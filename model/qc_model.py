from numpy import log
import torch
import torch.nn as nn
from .utils import *
from torch.nn import Parameter
from .utils import ACT2FN
from .layer_norms import LNorm

# =================================== 1. POSITIONAL ENCODING ======================================

# Given the embeddings for both the source and the target, the first step is to pass them through the
# positional encoding.
# REMARKS:
#   - The position encoding is designed so as the dot product of two positions p_i * p_{ i + t } is independent of t,
#     which has the desired effect that the architecture understands relative positions instead of absolute ones.
# Notice that this transformer may have problems with more than one encoder layers.
# The current PyTorch transformer can be used for more general cases.


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, config):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.positional_encoding_dropout)
        d_model = config.hidden_dim
        max_len = config.max_position_embeddings

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the embeddings
        :param x: (batch_size x seq_length x hidden_dim)
        :return: (batch_size x seq_length x hidden_dim) computes x + pe(x)
        """
        x = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# ============================= 2. ENCODER  AND ENCODER SUBLAYER ==================================================

# The encoder has a main class that wraps several layer of the transformer encoder sublayer.
# Remarks:
#   - The trick behind the attention mask is that it masks with large negative numbers that will become 0 after
#     passing by a Softmax.
# transformer_layrs
#   -EncoderLayer
#      -EncoderAttentionSublayer: MutlipleHeadAttention
#      -FeedForwardSublayer


class TransformerEncoder(nn.Module):

    def __init__(self, config):

        super(TransformerEncoder, self).__init__()
        self.transformer_layers = torch.nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.nb_encoder_layers)]
        )

    def forward(self, src_emb, look_ahead=False, device=torch.device("cpu")):
        """
        Encodes the given embeddings (
        :param src_emb: (batch_size x seq_length x hidden_dim) Embedding from source
        :param look_ahead: Bool, if the look ahead mask should be applied
        :param device: device("cuda") or device("cpu")
        :return: (batch_size x seq_length x hidden_dim) The so called contextual embeddings/encodings.
        """

        hidden_states = src_emb                                     # batch_size x seq_length x hidden_dim

        if look_ahead:
            look_ahead_mask = -1e9 * torch.triu(torch.ones(src_emb.size(1), src_emb.size(1)), diagonal=1)
        else:
            look_ahead_mask = torch.zeros(src_emb.size(1), src_emb.size(1))

        look_ahead_mask = look_ahead_mask.to(device)

        for transformer_layer in self.transformer_layers:

            hidden_states = transformer_layer(hidden_states, look_ahead_mask, device)
            # batch_size x seq_length x hidden_dim

        return hidden_states                                        # batch_size x seq_length x hidden_dim


# The encoder is made of several layers, each layer has an attention sublayer and a feed-forward part

class EncoderLayer(nn.Module):

    def __init__(self, config):

        super(EncoderLayer, self).__init__()

        self.attention_sublayer = EncoderAttentionSublayer(config)
        self.feed_forward_sublayer = FeedForwardSublayer(config)

    def forward(self, hidden_states, look_ahead_mask, device):
        """
        Outcome of one layer of the encoder
        :param hidden_states: (batch_size x seq_length x hidden_dim)
        :param look_ahead_mask: (seq_length x seq_length) The mask for look ahead
        :return: (batch_size x seq_length x hidden_dim)
        """
        attention_output = self.attention_sublayer(hidden_states, look_ahead_mask)
        # batch_size x seq_length x hidden_dim

        layer_output = self.feed_forward_sublayer(attention_output)     # batch_size x seq_length x hidden_dim

        return layer_output                                             # batch_size x seq_length x hidden_dim


# ================================ 2.a ENCODER SELF ATTENTION SUBLAYER ===============================================
# The attention sublayer computes the attention -> dense -> dropout -> add_inital (residual) -> LayerNormalization

class EncoderAttentionSublayer(nn.Module):

    def __init__(self, config):
        super(EncoderAttentionSublayer, self).__init__()
        self.multi_headed_attention = MultiHeadedAttention(config)
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.layer_norm = LNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor, look_ahead_mask):
        """
        Forward for the self attention sublayer of the encoder
        :param input_tensor: (batch_size x seq_length x hidden_dim)
        :param look_ahead_mask: (seq_length x seq_length) The mask for look ahead
        :return: (batch_size x seq_length x hidden_dim)
        """

        attention_output = self.multi_headed_attention(input_tensor,
                                                       input_tensor,
                                                       input_tensor,
                                                       look_ahead_mask) 
        hidden_states = self.dense(attention_output)                        # batch_size x seq_length x hidden_dim
        hidden_states = self.dropout(hidden_states)                         # batch_size x seq_length x hidden_dim

        attention_sublayer_output = \
            self.layer_norm(hidden_states + input_tensor)                   # batch_size x seq_length x hidden_dim

        return attention_sublayer_output                                    # batch_size x seq_length x hidden_dim


# The multi-headed attention "projects" to each head, different projection for each head. So to get
# The mixed_query, mixed_key, and mixed_value. each query and each query are multiplied and then normalized
# by the sqrt of the attention head size to reduce variance, so there is a score for each pair of entries
# a softmax is there applied to create probabilities and a dropout is applied (see note bellow)
# this probabilities vectors are multiplied by the mixed_value_layer to get an "interpolation"  of the
# different values. The projection are then concatenated to obtain the same dimension as started with.

class MultiHeadedAttention(nn.Module):

    def __init__(self, config):

        super(MultiHeadedAttention, self).__init__()

        if config.hidden_dim % config.num_attention_heads != 0:
            raise ValueError(
                """The hidden size {} is not a multiple of the number of attention "
                heads {}""".format(config.hidden_dim, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_dim / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_dim, self.all_head_size)
        self.key = nn.Linear(config.hidden_dim, self.all_head_size)
        self.value = nn.Linear(config.hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_scores(self, x):
        """
        An auxiliar method for transposing the results of for easy matrix multiplication
        :param x: (batch_size x seq_length x (num_heads * head_size))
        :return: (batch_size x num_heads x seq_length x head_size)
        """

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)
        
    def maskedsoftmax(self, x, look_ahead_mask):
        row_max = torch.max(x, -1)[0].unsqueeze(-1)
        exp_x = torch.exp(x - row_max)
        exp_x = exp_x * look_ahead_mask
        return exp_x  / torch.sum(exp_x, -1).unsqueeze(-1)

    def forward(self, query, key, value, look_ahead_mask):
        """
        Compute the multiheaded attention
        :param query: (batch_size x seq_length x hidden_dim)
        :param key: (batch_size x seq_length x hidden_dim)
        :param value: (batch_size x seq_length x hidden_dim)
        :param look_ahead_mask: (seq_length x seq_length) The mask for look ahead
        :return: (batch_size x seq_length x hidden_dim)
        """

        # num_head * head_size = hidden_dim = all_head_size

        mixed_query_layer = self.query(query)               # batch_size x seq_length x (num_heads * head_size)
        mixed_key_layer = self.key(key)                     # batch_size x seq_length x (num_heads * head_size)
        mixed_value_layer = self.value(value)               # batch_size x seq_length x (num_heads * head_size)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # batch_size x  num_heads x seq_length x head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)      # batch_size x  num_heads x seq_length x head_size
        value_layer = self.transpose_for_scores(mixed_value_layer)  # batch_size x  num_heads x seq_length x head_size

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # batch_size x  num_heads x seq_length x seq_length

        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size).float())
        # batch_size x  num_heads x seq_length x seq_length

        attention_scores = attention_scores + look_ahead_mask.unsqueeze(0).unsqueeze(0)
        # batch_size x  num_heads x seq_length x seq_length

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)      # batch_size x num_heads x seq_length x seq_length
        
        attention_probs = self.dropout(attention_probs)             # batch_size x num_heads x seq_length x seq_length

        context_layer = torch.matmul(attention_probs, value_layer)  # batch_size x num_heads x seq_length x head_size

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # batch_size x seq_length x num_heads x  head_size

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(*new_context_layer_shape)     # batch_size x seq_length x  hidden_dim

        return context_layer                                             # batch_size x seq_length x  hidden_dim


# ====================================== 2.b Feed-forward Sublayer ==============================================

# Next we have the second part of the transformer encoder sublayer. This layer takes the result from the attention
# sublayer and pass it through a dense part as hidden_dim -> intermediate_dim -> act_fun -> intermedate_dim ->
# -> dropout -> add input (resnet) -> normalize Note that the choice of the activation function is a hyper-parameter


class FeedForwardSublayer(nn.Module):

    def __init__(self, config):
        super(FeedForwardSublayer, self).__init__()
        self.feed_forward = FeedForward(config)
        self.dense = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.layer_norm = LNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, attention_tensor):
        """
        Forward method for the feed-forward part
        :param attention_tensor: ( batch_size x seq_length x hidden_dim)
        :return:  batch_size x seq_length x hidden_dim
        """

        hidden_states = self.feed_forward(attention_tensor)                 # batch_size x seq_length x intermediate_dim
        hidden_states = self.dense(hidden_states)                           # batch_size x seq_length x hidden_dim
        hidden_states = self.dropout(hidden_states)                         # batch_size x seq_length x hidden_dim
        layer_output = self.layer_norm(hidden_states + attention_tensor)     # batch_size x seq_length x hidden_dim
        return layer_output


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.dense = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        """
        :param hidden_states: ( batch_size x seq_length x hidden_dim)
        :return:  batch_size x seq_length x intermediate_dim
        """

        hidden_states = self.dense(hidden_states)                   # batch_size x seq_length x intermediate_dim
        hidden_states = self.intermediate_act_fn(hidden_states)     # batch_size x seq_length x intermediate_dim
        return hidden_states


# ========================================= MAIN MODEL ==========================================

# hidden_dim determines all the hidden layer size
# num_encoder_layer will apply transformer several times
class QCModel(nn.Module):
    def __init__(self, config, bias=None,
                 sampling_batch_size=10_000, sample_batch=10_000, nb_qbits=2, eval_nb_samples=100_000):
        super(QCModel, self).__init__()
        self.name = 'QCModel'
        self.config = config
        self.nb_measurements = config.nb_measurements
        # Model submodules
        # Note that we need an extra "word" for SOS
        self.emb = nn.Embedding(config.nb_measurements + 1, config.hidden_dim)
        self.positional_encoding = PositionalEncoding(config)
        self.transformer = TransformerEncoder(config)

        self.ff = nn.Linear(config.hidden_dim, config.nb_measurements)
        nn.init.zeros_(self.ff.weight)
        if bias is not None:
            self.ff.bias = Parameter(bias.type_as(self.ff.bias))
        else:
            self.ff.bias = nn.init.zeros_(self.ff.bias)

        # Auxiliary
        self.one_hot_emb_ = nn.Embedding(config.nb_measurements, config.nb_measurements)
        self.one_hot_emb_.weight.data = torch.eye(config.nb_measurements)
        self.one_hot_emb_.weight.data.requires_grad = False

        self.eval_nb_samples = eval_nb_samples
        self.sampling_batch_size = sampling_batch_size
        self.sample_batch_total = sample_batch
        self.nb_qbits = nb_qbits

    def device(self):

        return self.emb.weight.device

    def forward(self, forward_type='normal', seq=None,  look_ahead=False):
        """

        :param seq: tensor.long of size (nb_samples x nb_qbits).  Sequence of states? to be fed to the transformer
        :param look_ahead: Bool. Determines if mask the attention so only doing self attention to previous qbits
        :param device: device where the computation is running. (This can be made better)
        :return: tensor of size (nb_sample, nb_qbits, nb_measurements) corresponding to the pre-softmax probabilities of each state?
        """


        if forward_type == 'normal':
            seq_emb = self.emb(seq)
            seq_emb = self.positional_encoding(seq_emb)
            device = seq_emb.device
            trans = self.transformer(seq_emb, look_ahead, device)
            return self.ff(trans)

        elif forward_type == "sample":

            return self.sample_batch()

        elif forward_type == "evaluate":

            return self.sample()

        elif forward_type == "logP":
            return self.logP(seq)
        
        elif forward_type == 'sum_log_p':
            return self.sum_log_p(seq)
        
        elif forward_type == "name":
            return self.name

        else:
            raise NameError("{} not supported".format(forward_type))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_params_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def logP(self, seq, look_ahead=True):
        """
        Returns the distribution for the last qbit
        :param seq: tensor.long of size (nb_samples x nb_qbits).  Sequence of states? to be fed to the transformer
        :param look_ahead: Bool. Determines if mask the attention so only doing self attention to previous qbits
        :param device: device where the computation is running. (This can be made better)
        :return: (nb_samples, 1) for all qbits.
        """

        self.eval()
        device = seq.device
        nb_measurements = self.nb_measurements
        nb_samples, nb_qbits = seq.size()

        # Given a sample a_0, ..., a_n changes it to SOS, a_0, ..., a_{n-1}
        # nb_measurement = 4 is a different character for SOS
        init = nb_measurements * torch.ones((nb_samples, 1), dtype=torch.long, device=device)
        input_sample = torch.cat([init, seq], dim=1)[:, 0:nb_qbits]

        # The look_ahead mask is needed to only attend to previous qbits.
        probs = self.forward(forward_type="normal", seq=input_sample, look_ahead=True)  # n_samples x seq_len x nb_measurements
        log_p = torch.log_softmax(probs, dim=2)
        
        # Creates the one_hot_encodding
        eye = torch.eye(nb_measurements).to(device)
        one_hot = eye[seq]

        log_p = (one_hot * log_p).sum(dim=1).sum(dim=1)
        log_p = log_p.detach()
        self.train()

        return log_p
    
    def sum_log_p(self, seq):
        self.train()
        device = seq.device
        nb_measurements = self.nb_measurements
        nb_samples, nb_qbits = seq.size()

        # Given a sample a_0, ..., a_n changes it to SOS, a_0, ..., a_{n-1}
        # nb_measurement = 4 is a different character for SOS
        init = nb_measurements * torch.ones((nb_samples, 1), dtype=torch.long, device=device)
        input_sample = torch.cat([init, seq], dim=1)[:, 0:nb_qbits]

        # The look_ahead mask is needed to only attend to previous qbits.
        probs = self.forward(forward_type="normal", seq=input_sample, look_ahead=True)  # n_samples x seq_len x nb_measurements
        log_p = torch.log_softmax(probs, dim=2)
        
        # Creates the one_hot_encodding
        eye = torch.eye(nb_measurements).to(device)
        one_hot = eye[seq]

        log_p = (one_hot * log_p).sum(dim=1).sum(dim=1)

        return log_p
        

    def last_probs(self, seq, look_ahead=False):
        """
        Returns the distribution for the last qbit
        :param seq: tensor.long of size (nb_samples x nb_qbits).  Sequence of states? to be fed to the transformer
        :param look_ahead: Bool. Determines if mask the attention so only doing self attention to previous qbits
        :param device: device where the computation is running. (This can be made better)
        :return: (nb_samples, nb_measurements) distribution for last qbit.
        """

        logits = self.forward(seq=seq, look_ahead=look_ahead)

        return torch.log_softmax(logits[:, -1, :], dim=1)

    def sample(self):
        """
        Obtains a collection of samples and their log prob for a given number of qbits by batch_size at the time
        :param nb_samples: Desired number of samples
        :param sampling_batch_size: sampling batch size
        :param nb_qbits: Int, the number of qbits in the circuit
        :param device: device where the computation is running. (This can be made better)
        :return: torch.long, torch.float (cpu) of sizes (nb_samples, nb_qbits), (nb_samples, 1)
            where each long is in [0, nb_measurements).
        """

        number_calls = self.eval_nb_samples// self.sampling_batch_size
        samples = torch.zeros((self.eval_nb_samples , self.nb_qbits), dtype=torch.long)
        log_probs = torch.zeros((self.eval_nb_samples , 1))

        for j in range(number_calls):
            s, lp = self.sample_batch()
            lp = lp.unsqueeze(1)
            samples[j * self.sampling_batch_size: (j + 1) * self.sampling_batch_size] = s.detach().cpu()
            log_probs[j * self.sampling_batch_size: (j + 1) * self.sampling_batch_size] = lp.detach().cpu()

        if self.eval_nb_samples % self.sampling_batch_size:
            s, lp = self.sample_batch()
            lp = lp.unsqueeze(1)
            samples[number_calls * self.sampling_batch_size:] = s.detach().cpu()
            log_probs[number_calls * self.sampling_batch_size:] = lp.detach().cpu()

        return samples, log_probs.squeeze()

    def sample_batch(self):
        """
        Obtains a collection of samples for a given number of qbits associated to the current state of the model.
        :param sampling_batch_size: Int, the desired number of samples in the batch
        :param nb_qbits: Int, the number of qbits in the circuit
        :param device: device where the computation is running. (This can be made better)
        :return: torch.long, torch.float (cpu) of sizes (sampling_batch_size, nb_qbits), (sampling_batch_size, 1)
            where each long is in [0, nb_measurements).
        """

        device = self.device()
        self.eval()
        # Initialized
        output = self.config.nb_measurements * torch.ones((self.sampling_batch_size, 1), dtype=torch.long, device=device)
        log_probs = torch.zeros((self.sampling_batch_size), device=device)

        for i in range(self.nb_qbits):
            log_p = self.last_probs(output, False).detach()  # Batch_size x nb_meassurements
            probs = torch.exp(log_p)
            # We select the last qbit and sample from it
            pred_id = torch.multinomial(probs, 1, replacement=True)

            log_probs_temp = self.one_hot_emb_(pred_id).squeeze()

            log_probs_temp = log_probs_temp * log_p
            
            log_probs += log_probs_temp.sum(dim=1)

            output = torch.cat([output, pred_id], dim=1)

        output = output[:, 1:]
        log_probs = log_probs.detach()

        self.train()
        return output, log_probs
    
    
class StringModel(nn.Module):
    def __init__(self, config, bias=None,
                 sampling_batch_size=10_000, sample_batch=10_000, 
                 nb_qbits=4, nb_rows=2, nb_columns=2, 
                 eval_nb_samples=100_000, string_types=None, string_functions=None):
        super(StringModel, self).__init__()
        self.name = 'StringModel'
        self.config = config
        self.nb_measurements = config.nb_measurements
        # Model submodules
        # Note that we need an extra "word" for SOS
        self.emb = nn.Embedding(config.nb_measurements + 1, config.hidden_dim)
        self.positional_encoding = PositionalEncoding(config)
        self.transformer = TransformerEncoder(config)

        self.ff = nn.Linear(config.hidden_dim, config.nb_measurements)
        nn.init.zeros_(self.ff.weight)
        if bias is not None:
            self.ff.bias = Parameter(bias.type_as(self.ff.bias))
        else:
            self.ff.bias = nn.init.zeros_(self.ff.bias)


        # Auxiliary
        self.one_hot_emb_ = nn.Embedding(config.nb_measurements, config.nb_measurements)
        self.one_hot_emb_.weight.data = torch.eye(config.nb_measurements)
        self.one_hot_emb_.weight.data.requires_grad = False

        self.eval_nb_samples = eval_nb_samples
        self.sampling_batch_size = sampling_batch_size
        self.sample_batch_total = sample_batch
        self.nb_qbits = nb_qbits
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.string_types = string_types
        self.string_functions = string_functions
        
    def convert_random(self, seq, inverse = False):
        if self.string_functions is not None:
            nb_types = len(self.string_functions)
            random_types = torch.multinomial(torch.ones(nb_types), seq.shape[0], True)
            return torch.cat([self.string_functions[i][int(inverse)](seq[torch.where(random_types == i)]) for i in range(nb_types)])
        elif self.string_types is not None:
            nb_types = len(self.string_types)
            random_types = torch.multinomial(torch.ones(nb_types), seq.shape[0], True)
            return torch.cat([self.convert(seq[torch.where(random_types == i)], *self.string_types[i], inverse) for i in range(nb_types)])
        else:
            assert False, 'please provide a string'
    
    def convert_all(self, seq, inverse = False):
        if self.string_functions is not None:
            return torch.cat([string_function[int(inverse)](seq) for string_function in self.string_functions])
        elif self.string_types is not None:
            return torch.cat([self.convert(seq, *string_type, inverse) for string_type in self.string_types])
        else:
            assert False, 'please provide a string'
    
        
    def convert(self, seq, corner, vertical = False, inverse = False):
        """
        convert the input sequence
        :param seq: the input sequence of shape (batch size, self.nb_rows * self.nb_columns)
        :param method: which string to use
        :param inverse: apply the function in the inverse order or not
        :return: the converted sequence of shape (batch size, self.nb_rows * self.nb_columns)
        """
        if corner % 2 == 0:
            transpose = vertical
        else:
            transpose = not vertical
        if inverse == False:
            return self.string_normal(seq, corner, transpose)
        else:
            return self.string_inverse(seq, corner, transpose)
        
    
    def string_normal(self, seq, corner, transpose = False):
        """
        convert the input
        :return: the converted sequence
        """
        seq2d = seq.view(-1, self.nb_rows, self.nb_columns).clone()
        seq2d = seq2d.rot90(corner, [1, 2])
        if transpose:
            seq2d = seq2d.transpose(1, 2)
        seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
        return seq2d.contiguous().view(-1, self.nb_rows * self.nb_columns)
    
    def string_inverse(self, seq, corner, transpose = False):
        """
        convert back
        :return: the converted-back sequence
        """
        seq2d = seq.view(-1, self.nb_rows, self.nb_columns).clone()
        seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
        if transpose:
            seq2d = seq2d.transpose(1, 2)
        seq2d = seq2d.rot90(-corner, [1, 2])
        return seq2d.contiguous().view(-1, self.nb_rows * self.nb_columns)

    def device(self):

        return self.emb.weight.device

    def forward(self, forward_type='normal', seq=None,  look_ahead=False):
        """

        :param seq: tensor.long of size (nb_samples x nb_qbits).  Sequence of states? to be fed to the transformer
        :param look_ahead: Bool. Determines if mask the attention so only doing self attention to previous qbits
        :param device: device where the computation is running. (This can be made better)
        :return: tensor of size (nb_sample, nb_qbits, nb_measurements) corresponding to the pre-softmax probabilities of each state?
        """


        if forward_type == 'normal':
            seq_emb = self.emb(seq)
            seq_emb = self.positional_encoding(seq_emb)
            device = seq_emb.device
            trans = self.transformer(seq_emb, look_ahead, device)

            return self.ff(trans)

        elif forward_type == "sample":

            return self.sample_batch()

        elif forward_type == "evaluate":

            return self.sample()

        elif forward_type == "logP":
            return self.logP(seq)
        
        elif forward_type == 'sum_log_p':
            return self.sum_log_p(seq)
        
        elif forward_type == "name":
            return self.name

        else:
            raise NameError("{} not supported".format(forward_type))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_params_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def logP(self, seq, look_ahead=True):
        """
        Returns the distribution for the last qbit
        :param seq: tensor.long of size (nb_samples x nb_qbits).  Sequence of states? to be fed to the transformer
        :param look_ahead: Bool. Determines if mask the attention so only doing self attention to previous qbits
        :param device: device where the computation is running. (This can be made better)
        :return: (nb_samples, 1) for all qbits.
        """

        self.eval()
        device = seq.device
        nb_strings = len(self.string_functions) if self.string_functions is not None else len(self.string_types)
        all_seq = self.convert_all(seq)
        nb_measurements = self.nb_measurements
        nb_samples, nb_qbits = all_seq.size()

        # Given a sample a_0, ..., a_n changes it to SOS, a_0, ..., a_{n-1}
        # nb_measurement = 4 is a different character for SOS
        init = nb_measurements * torch.ones((nb_samples, 1), dtype=torch.long, device=device)
        input_sample = torch.cat([init, all_seq], dim=1)[:, 0:nb_qbits]

        # The look_ahead mask is needed to only attend to previous qbits.
        probs = self.forward(forward_type="normal", seq=input_sample, look_ahead=True)  # n_samples x seq_len x nb_measurements
        log_p = torch.log_softmax(probs, dim=2)

        # Creates the one_hot_encodding
        eye = torch.eye(nb_measurements).to(device)
        one_hot = eye[all_seq]

        log_p = (one_hot * log_p).sum(dim=1).sum(dim=1)
        log_p = log_p.view(nb_strings, -1)
        log_p = torch.logsumexp(log_p, 0)-log(nb_strings)
        log_p = log_p.detach()
        self.train()
        
        return log_p
    
    def sum_log_p(self, seq):
        self.train()
        device = seq.device
        nb_strings = len(self.string_functions) if self.string_functions is not None else len(self.string_types)
        all_seq = self.convert_all(seq)
        nb_measurements = self.nb_measurements
        nb_samples, nb_qbits = all_seq.size()

        # Given a sample a_0, ..., a_n changes it to SOS, a_0, ..., a_{n-1}
        # nb_measurement = 4 is a different character for SOS
        init = nb_measurements * torch.ones((nb_samples, 1), dtype=torch.long, device=device)
        input_sample = torch.cat([init, all_seq], dim=1)[:, 0:nb_qbits]

        # The look_ahead mask is needed to only attend to previous qbits.
        probs = self.forward(forward_type="normal", seq=input_sample, look_ahead=True)  # n_samples x seq_len x nb_measurements
        log_p = torch.log_softmax(probs, dim=2)
        # Creates the one_hot_encodding
        eye = torch.eye(nb_measurements).to(device)
        one_hot = eye[all_seq]

        log_p = (one_hot * log_p).sum(dim=1).sum(dim=1)
        log_p = log_p.view(nb_strings, -1)
        log_p = torch.logsumexp(log_p, 0)-log(nb_strings)
        
        return log_p
        

    def last_probs(self, seq, look_ahead=False):
        """
        Returns the distribution for the last qbit
        :param seq: tensor.long of size (nb_samples x nb_qbits).  Sequence of states? to be fed to the transformer
        :param look_ahead: Bool. Determines if mask the attention so only doing self attention to previous qbits
        :param device: device where the computation is running. (This can be made better)
        :return: (nb_samples, nb_measurements) distribution for last qbit.
        """

        logits = self.forward(seq=seq, look_ahead=look_ahead)

        return torch.log_softmax(logits[:, -1, :], dim=1)

    def sample(self):
        """
        Obtains a collection of samples and their log prob for a given number of qbits by batch_size at the time
        :param nb_samples: Desired number of samples
        :param sampling_batch_size: sampling batch size
        :param nb_qbits: Int, the number of qbits in the circuit
        :param device: device where the computation is running. (This can be made better)
        :return: torch.long, torch.float (cpu) of sizes (nb_samples, nb_qbits), (nb_samples, 1)
            where each long is in [0, nb_measurements).
        """

        number_calls = self.eval_nb_samples// self.sampling_batch_size
        samples = torch.zeros((self.eval_nb_samples , self.nb_qbits), dtype=torch.long)
        log_probs = torch.zeros((self.eval_nb_samples , 1))

        for j in range(number_calls):
            s, lp = self.sample_batch()
            lp = lp.unsqueeze(1)
            samples[j * self.sampling_batch_size: (j + 1) * self.sampling_batch_size] = s.detach().cpu()
            log_probs[j * self.sampling_batch_size: (j + 1) * self.sampling_batch_size] = lp.detach().cpu()

        if self.eval_nb_samples % self.sampling_batch_size:
            s, lp = self.sample_batch()
            lp = lp.unsqueeze(1)
            samples[number_calls * self.sampling_batch_size:] = s.detach().cpu()
            log_probs[number_calls * self.sampling_batch_size:] = lp.detach().cpu()

        return samples, log_probs.squeeze()

    def sample_batch(self):
        """
        Obtains a collection of samples for a given number of qbits associated to the current state of the model.
        :param sampling_batch_size: Int, the desired number of samples in the batch
        :param nb_qbits: Int, the number of qbits in the circuit
        :param device: device where the computation is running. (This can be made better)
        :return: torch.long, torch.float (cpu) of sizes (sampling_batch_size, nb_qbits), (sampling_batch_size, 1)
            where each long is in [0, nb_measurements).
        """

        device = self.device()
        self.eval()
        # Initialized
        output = self.config.nb_measurements * torch.ones((self.sampling_batch_size, 1), dtype=torch.long, device=device)

        for i in range(self.nb_qbits):
            log_p = self.last_probs(output, False).detach()  # Batch_size x nb_meassurements
            probs = torch.exp(log_p)
            # We select the last qbit and sample from it
            pred_id = torch.multinomial(probs, 1, replacement=True)

            output = torch.cat([output, pred_id], dim=1)

        output = output[:, 1:]
        output = self.convert_random(output, inverse = True)
        log_probs = self.logP(output)

        self.train()
        return output, log_probs
