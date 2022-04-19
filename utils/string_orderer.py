import torch
import numpy as np


class StringOrderer1d:
    def __init__(self, nb_qbit, device = torch.device('cpu')):
        self.nb_qbit = nb_qbit
        self.device = device
        
    def __call__(self, left = None, right = None, start_at = 'left', string_inverse_pair = None):
        if string_inverse_pair is None:
            if start_at == 0 or start_at == 'left':
                stride = 1
            elif start_at == 1 or start_at == 'right':
                stride = -1
            else:
                assert False, 'please select between left (0), and right (1)'
            
            
            if left is None and right is None:
                left = 0
            elif left is not None and right is not None:
                assert False, 'please specify only one end'
            elif right is not None:
                left = (right+1) % self.nb_qbit
                
            str_ind = self.periodic_translate(np.arange(self.nb_qbit), left)[::stride]
            str_inv = self.periodic_translate_inverse(np.arange(self.nb_qbit)[::stride], left, stride)
            
        else:
            str_ind, str_inv = string_inverse_pair
            
        str_ind = torch.from_numpy(np.ascontiguousarray(str_ind)).to(self.device).long()
        str_inv = torch.from_numpy(np.ascontiguousarray(str_inv)).to(self.device).long()
        return lambda seq: seq[:, str_ind].clone(), lambda seq: seq[:, str_inv].clone()
        
    def periodic_translate(self, ind, left):
        return (ind + left) % self.nb_qbit
    
    def periodic_translate_inverse(self, ind, left, stride):
        return (ind - stride * left) % self.nb_qbit
        
    
    
class StringOrderer2d:
    def __init__(self, nb_rows, nb_columns, device = torch.device('cpu')):
        """
        :param nb_rows: the total number of rows
        :param nb_columns: the total number of columns
        :param string_types: list of tuples of the format [(corner, vertical)]
        :param seq_size: the size of the sequence 
        """
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.device = device
        
    def __call__(self, top_left = None, top_right = None, buttom_left = None, buttom_right = None, start_at = 'top_left', vertical = False, string_inverse_pair = None):
        """
        :return: a function to convert a sequence according to a specific string and its inverse function.
        """

        if string_inverse_pair is None:
            
                    
            # get start location
            if start_at == 0 or start_at == 'top_left':
                corner = 0
            elif start_at == 1 or start_at == 'top_right':
                corner = 1
            elif start_at == 2 or start_at == 'buttom_right':
                corner = 2
            elif start_at == 3 or start_at == 'buttom_left':
                corner = 3
            else:
                assert False, 'please select between top_left (0), top_right (1), buttom_left (3), and buttom_right (2)'
            
            # make sure only one corner is specified
            if [top_left, top_right, buttom_left, buttom_right].count(None) < 3:
                assert False, 'please specify only one corner'
            elif [top_left, top_right, buttom_left, buttom_right].count(None) == 4:
                top_left = 0
            elif not top_right is None:
                r, c = self.number_to_point(top_right)
                top_left = self.point_to_number(r, (c+1)%self.nb_columns)
            elif not buttom_left is None:
                r, c = self.number_to_point(buttom_left)
                top_left = self.point_to_number((r+1)%self.nb_rows, c)
            elif not buttom_right is None:
                r, c = self.number_to_point(buttom_right)
                top_left = self.point_to_number((r+1)%self.nb_rows, (c+1)%self.nb_columns)
            
            # the index and inverse index
            if top_left == 0:
                ind2d = np.arange(self.nb_rows*self.nb_columns).reshape(self.nb_rows, self.nb_columns)
                str_ind = self.convert(ind2d, corner, vertical, False).reshape(self.nb_rows * self.nb_columns)
                if vertical:
                    ind2d = np.arange(self.nb_rows*self.nb_columns).reshape(self.nb_columns, self.nb_rows)
                else:
                    ind2d = np.arange(self.nb_rows*self.nb_columns).reshape(self.nb_rows, self.nb_columns)
                str_inv = self.convert(ind2d, corner, vertical, True).reshape(self.nb_rows * self.nb_columns)
            else:
                ind2d = np.arange(self.nb_rows*self.nb_columns).reshape(self.nb_rows, self.nb_columns)
                ind2d = self.periodic_translate(ind2d, top_left)
                str_ind = self.convert(ind2d, corner, vertical, False).reshape(self.nb_rows * self.nb_columns)
                if vertical:
                    ind2d = np.arange(self.nb_rows*self.nb_columns).reshape(self.nb_columns, self.nb_rows)
                else:
                    ind2d = np.arange(self.nb_rows*self.nb_columns).reshape(self.nb_rows, self.nb_columns)
                str_inv = self.convert(ind2d, corner, vertical, True)
                str_inv = self.periodic_translate_inverse(str_inv, top_left).reshape(self.nb_rows * self.nb_columns)
        else:
            str_ind, str_inv = string_inverse_pair
        str_ind = torch.from_numpy(np.ascontiguousarray(str_ind)).to(self.device).long()
        str_inv = torch.from_numpy(np.ascontiguousarray(str_inv)).to(self.device).long()
        return lambda seq: seq[:, str_ind].clone(), lambda seq: seq[:, str_inv].clone()
        
    
    def periodic_translate(self, ind2d, top_left):
        """
        :param top_left: the location of the top_left corner
        :return: the translated 2darray
        """
        rows, columns = ind2d.shape
        r, c = self.number_to_point(top_left)
        pad_width = np.maximum(rows, columns)
        ind2d = np.pad(ind2d, pad_width, 'wrap')
        r = r + pad_width
        c = c + pad_width
        return ind2d[r:r+rows, c:c+columns]
    
    def periodic_translate_inverse(self, ind2d, top_left):
        """
        :param top_left: the location of the top_left corner
        :return: the translated-back 2darray
        """
        rows, columns = ind2d.shape
        r, c = self.number_to_point(top_left)
        pad_width = np.maximum(rows, columns)
        ind2d = np.pad(ind2d, pad_width, 'wrap')
        r = pad_width - r
        c = pad_width - c
        return ind2d[r:r+rows, c:c+columns]
        
        
    
    def point_to_number(self, r_num, c_num):
        """
        covert the 2d point into a 1d number according to the sequence
        0 1 2
        3 4 5
        6 7 8
        :param r_num: the row number (first index) of the 2d point
        :param c_num: the column number (second index) of the 2d point
        :param num_columns: the total number of columns
        :return: the corresponding 1d number
        """
        return r_num * self.nb_columns + c_num

    def number_to_point(self, num):
        """
        covert the 1d number into a 2d point according to the sequence
        0 1 2
        3 4 5
        6 7 8
        :param num: 1d number
        :param num_columns: the total number of columns
        :return: the corresponding 2d point in the order (row number, column number)
        """
        return num // self.nb_columns, num % self.nb_columns
    
    def convert(self, seq, corner, vertical = False, inverse = False):
        """
        convert the input sequence to one of the following strings
        corner:        vertical:         string:
               0                False           1
               0                True            2
               1                False           4
               1                True            3
               2                False           5
               2                True            6
               3                False           8
               3                True            7
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
            return self.normal(seq, corner, transpose)
        else:
            return self.inverse(seq, corner, transpose)
        
    def normal(self, ind2d, corner, transpose = False):
        """
        convert the input
        :return: the converted sequence
        """
        ind2d = np.rot90(ind2d, corner, [0, 1])
        if transpose:
            ind2d = ind2d.T
        ind2d[1::2, :] = ind2d[1::2, ::-1]
        return ind2d
    
    def inverse(self, ind2d, corner, transpose = False):
        """
        convert back
        :return: the converted-back sequence
        """
        ind2d[1::2, :] = ind2d[1::2, ::-1]
        if transpose:
            ind2d = ind2d.T
        ind2d = np.rot90(ind2d, -corner, [0, 1])
        return ind2d
    
    
    def string1(self, seq, inverse = False):
        """
        convert a given sequence in the order
        0 1 2  to  0 1 2  or  *-> ->  |
        3 4 5      5 4 3       |  <- <-
        6 7 8      6 7 8       -> -> ->#
        :param seq: the input sequence of shape (batch size, self.nb_rows * self.nb_columns)
        :return: the converted sequence of shape (batch size, self.nb_rows * self.nb_columns)
        """
        seq2d = seq.view(-1, self.nb_rows, self.nb_columns).clone()
        seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
        return seq2d.contiguous().view(-1, self.nb_rows * self.nb_columns)
        
    
    def string2(self, seq, inverse = False):
        """
        convert a given sequence in the order
        0 1 2  to  0 5 6  or  *| ->  |
        3 4 5      1 4 7       |  |  |
        6 7 8      2 3 8       -> |  |#
        :param seq: the input sequence of shape (batch size, self.nb_rows * self.nb_columns)
        :return: the converted sequence of shape (batch size, self.nb_rows * self.nb_columns)
        """
        seq2d = seq.view(-1, self.nb_rows, self.nb_columns).clone()
        if inverse == False:
            seq2d = seq2d.transpose(1, 2)
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
        else:
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
            seq2d = seq2d.transpose(1, 2)
        return seq2d.contiguous().view(-1, self.nb_rows * self.nb_columns)
    
    def string3(self, seq, inverse = False):
        """
        convert a given sequence in the order
        0 1 2  to  6 5 0  or   | <-  |*
        3 4 5      7 4 1       |  |  |
        6 7 8      8 3 2      #|  | <-
        :param seq: the input sequence of shape (batch size, self.nb_rows * self.nb_columns)
        :return: the converted sequence of shape (batch size, self.nb_rows * self.nb_columns)
        """
        seq2d = seq.view(-1, self.nb_rows, self.nb_columns).clone()
        if inverse == False:
            seq2d = seq2d.rot90(1, [1, 2])
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
        else:
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
            seq2d = seq2d.rot90(-1, [1, 2])
        return seq2d.contiguous().view(-1, self.nb_rows * self.nb_columns)
    
    def string4(self, seq, inverse = False):
        """
        convert a given sequence in the order
        0 1 2  to  2 1 0  or   |  <- <-*
        3 4 5      3 4 5       -> ->  |
        6 7 8      8 7 6      #<- <- <-
        :param seq: the input sequence of shape (batch size, self.nb_rows * self.nb_columns)
        :return: the converted sequence of shape (batch size, self.nb_rows * self.nb_columns)
        """
        seq2d = seq.view(-1, self.nb_rows, self.nb_columns).clone()
        if inverse == False:
            seq2d = seq2d.rot90(1, [1, 2]).transpose(1, 2)
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
        else:
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
            seq2d = seq2d.transpose(1, 2).rot90(-1, [1, 2])
        return seq2d.contiguous().view(-1, self.nb_rows * self.nb_columns)
    
    def string5(self, seq, inverse = False):
        """
        convert a given sequence in the order
        0 1 2  to  8 7 6  or  #<- <- <-
        3 4 5      3 4 5       -> ->  |
        6 7 8      2 1 0       |  <- <-*
        :param seq: the input sequence of shape (batch size, self.nb_rows * self.nb_columns)
        :return: the converted sequence of shape (batch size, self.nb_rows * self.nb_columns)
        """
        seq2d = seq.view(-1, self.nb_rows, self.nb_columns).clone()
        if inverse == False:
            seq2d = seq2d.rot90(2, [1, 2])
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
        else:
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
            seq2d = seq2d.rot90(-2, [1, 2])
        return seq2d.contiguous().view(-1, self.nb_rows * self.nb_columns)
    
    def string6(self, seq, inverse = False):
        """
        convert a given sequence in the order
        0 1 2  to  8 3 2  or  #|  | <-
        3 4 5      7 4 1       |  |  |
        6 7 8      6 5 0       | <-  |*
        :param seq: the input sequence of shape (batch size, self.nb_rows * self.nb_columns)
        :return: the converted sequence of shape (batch size, self.nb_rows * self.nb_columns)
        """
        seq2d = seq.view(-1, self.nb_rows, self.nb_columns).clone()
        if inverse == False:
            seq2d = seq2d.rot90(2, [1, 2]).transpose(1, 2)
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
        else:
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
            seq2d = seq2d.transpose(1, 2).rot90(-2, [1, 2])
        return seq2d.contiguous().view(-1, self.nb_rows * self.nb_columns)
    
    def string7(self, seq, inverse = False):
        """
        convert a given sequence in the order
        0 1 2  to  2 3 8  or   -> |  |#
        3 4 5      1 4 7       |  |  |
        6 7 8      0 5 6      *|  -> |
        :param seq: the input sequence of shape (batch size, self.nb_rows * self.nb_columns)
        :return: the converted sequence of shape (batch size, self.nb_rows * self.nb_columns)
        """
        seq2d = seq.view(-1, self.nb_rows, self.nb_columns).clone()
        if inverse == False:
            seq2d = seq2d.rot90(3, [1, 2])
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
        else:
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
            seq2d = seq2d.rot90(-3, [1, 2])
        return seq2d.contiguous().view(-1, self.nb_rows * self.nb_columns)
    
    def string8(self, seq, inverse = False):
        """
        convert a given sequence in the order
        0 1 2  to  6 7 8  or   -> -> ->#
        3 4 5      5 4 3       |  <- <-
        6 7 8      0 1 2      *-> ->  |
        :param seq: the input sequence of shape (batch size, self.nb_rows * self.nb_columns)
        :return: the converted sequence of shape (batch size, self.nb_rows * self.nb_columns)
        """
        seq2d = seq.view(-1, self.nb_rows, self.nb_columns).clone()
        if inverse == False:
            seq2d = seq2d.rot90(3, [1, 2]).transpose(1, 2)
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
        else:
            seq2d[:, 1::2, :] = seq2d[:, 1::2, :].flip(2)
            seq2d = seq2d.transpose(1, 2).rot90(-3, [1, 2])
        return seq2d.contiguous().view(-1, self.nb_rows * self.nb_columns)