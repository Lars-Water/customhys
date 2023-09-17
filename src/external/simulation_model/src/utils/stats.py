import time

class Stats:
    stats = None

    def __init__(self, pre_stats=None):
        if pre_stats and isinstance(pre_stats, dict):
            self.stats = pre_stats
        else:
            self.stats = {}

    @staticmethod
    def merge_dicts(dict1, dict2, path=None):
        # based on: https://stackoverflow.com/a/7205107
        if path is None: path = []
        for key in dict2:
            if key in dict1:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    Stats.merge_dicts(dict1[key], dict2[key], path + [str(key)])
                elif dict1[key] == dict2[key]:
                    pass # same leaf value
                else:
                    raise Exception("Conflict at {}".format(".".join(path + [str(key)])))
            else:
                dict1[key] = dict2[key]
        return dict1

    @staticmethod
    def update_dict(current_dict, update_dict):
        test = {}
        for k, v in update_dict.items():
            if isinstance(v, dict):
                current_dict[k] = Stats.update_dict(current_dict.get(k, {}), v)
            else:
                current_dict[k] = v

        return current_dict

    @staticmethod
    def create_dict_from_args_val(val, *args):
        cur_dict = {}
        for i, arg in enumerate(reversed(args)):
            if i == 0:
                cur_dict = {arg: val}
            else:
                cur_dict = {arg: cur_dict}

        return cur_dict

    def has_stat(self, *args):
        cur = self.stats
        for arg in args:
            if arg in cur:
                cur = cur[arg]
            else:
                return False

        if isinstance(cur, dict):
            return False
        else:
            return True

    def get_stat(self, *args):
        if self.has_stat(*args):
            cur = self.stats
            for arg in args:
                cur = cur[arg]
            return cur
        else:
            return None

    def record_stat(self, val, *args):
        merge_dict = Stats.create_dict_from_args_val(val, *args)
        self.stats = Stats.update_dict(self.stats, merge_dict)

    def record_time_stat(self, *args):
        record_time = time.time()
        self.record_stat(record_time, *args)
        return record_time

    def inc_stat(self, *args):
        self.add_stat(1, *args)

    def add_stat(self, val, *args):
        old_val = self.get_stat()
        new_val = val

        if old_val:
            new_val = val + old_val

        self.record_stat(new_val, *args)

    def get_stats_dict(self):
        return self.stats