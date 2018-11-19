import os
import torch
import json
from copy import deepcopy
import glob2
import parse
import shutil
class Saver(object):
    def __init__(self, model_dir, max_to_keep=5):
        self.model_dir = model_dir
        self.max_to_keep = max_to_keep

    def list_model_dir(self):
        model_list_path = os.path.join(self.model_dir, 'model_list.json')
        #if not os.path.exists(model_list_path):
        t_model_list = sorted(glob2.glob(os.path.join(self.model_dir, 'model_*.npy')))
        t_model_list = [os.path.basename(t) for t in t_model_list]
        step_list = [parse.parse('model_{:06d}.npy', t)[0] for t in t_model_list]
        model_list = [dict(step=step_list[t], model_path=t_model_list[t]) for t in range(len(t_model_list))]
        model_list = sorted(model_list, key=lambda k: k['step'])
        return model_list

    def save(self, model, info_dict, step, is_best=False):
        print('saving...')
        model_list = self.list_model_dir()

        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model_f = model#.float().eval()

        if len(model_list) > self.max_to_keep:
            os.remove(os.path.join(self.model_dir, model_list[0]['model_path']))

            del model_list[0]
        state = dict(model_state_dict=model_f.state_dict())
        state.update(info_dict)
        step_str = str(step)
        model_path = os.path.join(self.model_dir, 'model_{}.npy'.format(step_str))
        torch.save(state, model_path)
        model_list.append(dict(step=step, model_path='model_{}.npy'.format(step_str)))
        with open(os.path.join(self.model_dir, 'model_list.json'), 'w') as f:
            json.dump(model_list, f, indent=2)

        if is_best:
            shutil.copyfile(model_path, os.path.join(self.model_dir, 'model_best.pth.tar'))

    def load_latest(self):
        model_list = self.list_model_dir()
        if len(model_list) == 0:
            return None
        print(model_list[-1])
        check_point = torch.load(os.path.join(self.model_dir, model_list[-1]['model_path']))

        return check_point

    def load_best(self):
        if os.path.exists(os.path.join(self.model_dir, 'model_best.pth.tar')):
            check_point = torch.load(os.path.join(self.model_dir, 'model_best.pth.tar'))
            return check_point
        return None

def make_log_dirs(dump_basedir, run_id):
    log_dir = os.path.join(dump_basedir, run_id)
    train_dir = os.path.join(log_dir, 'train')
    test_dir = os.path.join(log_dir, 'test')
    test_dir = os.path.join(log_dir, 'test_ap')  # delete this line when not needed
    model_dir = os.path.join(log_dir, 'model')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir, train_dir, log_dir