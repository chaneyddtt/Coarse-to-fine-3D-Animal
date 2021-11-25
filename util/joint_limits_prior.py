import numpy as np
import torch
Ranges = {
    'pelvis': [[0, 0], [0, 0], [0, 0]],
    'pelvis0': [[-0.3, 0.3], [-1.2, 0.5], [-0.1, 0.1]],
    'spine': [[-0.4, 0.4], [-1.0, 0.9], [-0.8, 0.8]],
    'spine0': [[-0.4, 0.4], [-1.0, 0.9], [-0.8, 0.8]],
    'spine1': [[-0.4, 0.4], [-0.5, 1.2], [-0.4, 0.4]],
    'spine3': [[-0.5, 0.5], [-0.6, 1.4], [-0.8, 0.8]],
    'spine2': [[-0.5, 0.5], [-0.4, 1.4], [-0.5, 0.5]],
    'RFootBack': [[-0.2, 0.3], [-0.3, 1.1], [-0.3, 0.5]],
    'LFootBack': [[-0.3, 0.2], [-0.3, 1.1], [-0.5, 0.3]],
    'LLegBack1': [[-0.2, 0.3], [-0.5, 0.8], [-0.5, 0.4]],
    'RLegBack1': [[-0.3, 0.2], [-0.5, 0.8], [-0.4, 0.5]],
    'Head': [[-0.5, 0.5], [-1.0, 0.9], [-0.9, 0.9]],
    'RLegBack2': [[-0.3, 0.2], [-0.6, 0.8], [-0.5, 0.6]],
    'LLegBack2': [[-0.2, 0.3], [-0.6, 0.8], [-0.6, 0.5]],
    'RLegBack3': [[-0.2, 0.3], [-0.8, 0.2], [-0.4, 0.5]],
    'LLegBack3': [[-0.3, 0.2], [-0.8, 0.2], [-0.5, 0.4]],
    'Mouth': [[-0.1, 0.1], [-1.1, 0.5], [-0.1, 0.1]],
    'Neck': [[-0.8, 0.8], [-1.0, 1.0], [-1.1, 1.1]],
    'LLeg1': [[-0.05, 0.05], [-1.3, 0.8], [-0.6, 0.6]],  # Extreme
    'RLeg1': [[-0.05, 0.05], [-1.3, 0.8], [-0.6, 0.6]],
    'RLeg2': [[-0.05, 0.05], [-1.0, 0.9], [-0.6, 0.6]],  # Extreme
    'LLeg2': [[-0.05, 0.05], [-1.0, 1.1], [-0.6, 0.6]],
    'RLeg3': [[-0.1, 0.4], [-0.3, 1.4], [-0.4, 0.7]],  # Extreme
    'LLeg3': [[-0.4, 0.1], [-0.3, 1.4], [-0.7, 0.4]],
    'LFoot': [[-0.3, 0.1], [-0.4, 1.5], [-0.7, 0.3]],  # Extreme
    'RFoot': [[-0.1, 0.3], [-0.4, 1.5], [-0.3, 0.7]],
    'Tail7': [[-0.1, 0.1], [-0.7, 1.1], [-0.9, 0.8]],
    'Tail6': [[-0.1, 0.1], [-1.4, 1.4], [-1.0, 1.0]],
    'Tail5': [[-0.1, 0.1], [-1.0, 1.0], [-0.8, 0.8]],
    'Tail4': [[-0.1, 0.1], [-1.0, 1.0], [-0.8, 0.8]],
    'Tail3': [[-0.1, 0.1], [-1.0, 1.0], [-0.8, 0.8]],
    'Tail2': [[-0.1, 0.1], [-1.0, 1.0], [-0.8, 0.8]],
    'Tail1': [[-0.1, 0.1], [-1.5, 1.4], [-1.2, 1.2]],
}


class LimitPrior(object):
    def __init__(self, device, n_pose=32):
        self.parts = {
            'pelvis0': 0,
            'spine': 1,
            'spine0': 2,
            'spine1': 3,
            'spine2': 4,
            'spine3': 5,
            'LLeg1': 6,
            'LLeg2': 7,
            'LLeg3': 8,
            'LFoot': 9,
            'RLeg1': 10,
            'RLeg2': 11,
            'RLeg3': 12,
            'RFoot': 13,
            'Neck': 14,
            'Head': 15,
            'LLegBack1': 16,
            'LLegBack2': 17,
            'LLegBack3': 18,
            'LFootBack': 19,
            'RLegBack1': 20,
            'RLegBack2': 21,
            'RLegBack3': 22,
            'RFootBack': 23,
            'Tail1': 24,
            'Tail2': 25,
            'Tail3': 26,
            'Tail4': 27,
            'Tail5': 28,
            'Tail6': 29,
            'Tail7': 30,
            'Mouth': 31
        }
        self.id2name = {v: k for k, v in self.parts.items()}
        # Ignore the first joint.
        self.prefix = 3
        self.postfix= 99
        self.part_ids = np.array(sorted(self.parts.values()))
        min_values = np.hstack([np.array(np.array(Ranges[self.id2name[part_id]])[:, 0]) for part_id in self.part_ids])
        max_values = np.hstack([
            np.array(np.array(Ranges[self.id2name[part_id]])[:, 1])
            for part_id in self.part_ids
        ])
        self.ranges = Ranges
        self.device = device
        self.min_values = torch.from_numpy(min_values).view(n_pose, 3).float().to(device)
        self.max_values = torch.from_numpy(max_values).view(n_pose, 3).float().to(device)

    def __call__(self, x):
        '''
        Given x, rel rotation of 31 joints, for each parts compute the limit value.
        k is steepness of the curve, max_val + margin is the midpoint of the curve (val 0.5)
        Using Logistic:
        max limit: 1/(1 + exp(k * ((max_val + margin) - x)))
        min limit: 1/(1 + exp(k * (x - (min_val - margin))))
        With max/min:
        minlimit: max( min_vals - x , 0 )
        maxlimit: max( x - max_vals , 0 )
        With exponential:
        min: exp(k * (minval - x) )
        max: exp(k * (x - maxval) )
        '''
        ## Max/min discontinous but fast. (flat + L2 past the limit)

        x = x[:, self.prefix:self.postfix].view(x.shape[0], -1, 3)
        zeros = torch.zeros_like(x).to(self.device)
        # return np.maximum(x - self.max_values, zeros) + np.maximum(self.min_values - x, zeros)
        return torch.mean(torch.max(x - self.max_values.unsqueeze(0),  zeros) + torch.max(self.min_values.unsqueeze(0) - x, zeros))

    def report(self, x):
        res = self(x).r.reshape(-1, 3)
        values = x[self.prefix:].r.reshape(-1, 3)
        bad = np.any(res > 0, axis=1)
        bad_ids = np.array(self.part_ids)[bad]
        np.set_printoptions(precision=3)
        for bad_id in bad_ids:
            name = self.id2name[bad_id]
            limits = self.ranges[name]
            print('%s over! Overby:' % name),
            print(res[bad_id - 1, :]),
            print(' Limits:'),
            print(limits),
            print(' Values:'),
            print(values[bad_id - 1, :])

if __name__ == '__main__':
    name2id33 = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1,
                 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15,
                 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11,
                 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27,
                 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'root': 0}
    name2id35 = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1,
                 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15,
                 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11,
                 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27,
                 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'root': 0, 'LEar': 33, 'REar': 34}

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limit_prior = LimitPrior(device, 32)
    for k,v in limit_prior.parts.items():
        id33 = name2id33[k]-1
        id35 = name2id35[k]-1
        assert id33 == id35 and id33==v
    x = torch.zeros((35*3,)).float().to(device)
    limit_loss = limit_prior(x)
    print('done')