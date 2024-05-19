"""
ReplayBuffer class file.
"""

from collections import deque
import random


class ReplayBuffer:
    """
    Class of replay buffer.
    This replay buffer includes functions such as push and pull.
    Each data = state, action, advant, return, action_prob.
    """

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, data):
        """
        Push one set of data into buffer.

        :arg data:
            data should be a tuple of \
                state, action, advant, return, action_prob.
        """
        self.buffer.append(data)

    def pull(self, data_size):
        """
        Pull data of size data_size from buffer.

        :arg data_size:
            The size of data which will be pulled from buffer.

        :return:
            A tuple which consists of lists of \
                state, action, advant, return, action_prob.
        """
        minibatch = random.sample(self.buffer, data_size)
        s_lst, a_lst, adv_lst, ret_lst, op_lst, id_lst = [], [], [], [], [], []

        for s, a, adv, ret, op, agent_id in minibatch:
            s_lst.append(s)
            a_lst.append(a)
            adv_lst.append(adv)
            ret_lst.append(ret)
            op_lst.append(op)
            id_lst.append(agent_id)

        return s_lst, a_lst, adv_lst, ret_lst, op_lst, id_lst

    def size(self):
        """
        Returns the size of buffer.
        """
        return len(self.buffer)
