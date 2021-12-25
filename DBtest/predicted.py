from datetime import datetime


class Predict:

    def __init__(self):
        super(Predict, self).__init__()

        self.idx = 0
        self.interval = 0
        self.cur_stage = 0
        self.stage_count = 0
        self.last_stage_idx = -1
        self.end_time = self.stage_count * self.interval * 10
        self.last_time = 0

    def updates(self, stages, times):
        rests = []
        probabilities = []
        for i in range(len(stages)):
            self.update(stages[i], times[i])
            if stages[i] == 0:
                continue
            rest_time, probability = self.result(times[i])
            rests.append(rest_time)
            probabilities.append(probability)
        return rests, probabilities

    def update(self, stage, time):
        # 判断是否改变当前阶段
        w_a, w_b = 3, 1
        if 0 < stage - self.cur_stage < 2 and self.idx - self.last_stage_idx >= self.stage_count:
            self.cur_stage = stage

            self.stage_count = self.idx - (self.last_stage_idx + 1) \
                if self.stage_count == 0 else \
                int((w_a * self.stage_count + w_b * (self.idx - self.last_stage_idx - 1)) / (w_a + w_b))

            self.last_stage_idx = self.idx - 1

        self.idx += 1
        if self.last_time == 0:
            self.last_time = time
        self.interval = int((w_b * self.interval + w_a * (time - self.last_time)) / (w_a + w_b))
        self.last_time = time
        if not self.stage_count == 0:
            self.end_time = time + \
                            (self.stage_count - (self.idx - (self.last_stage_idx + 1))) * self.interval + \
                            self.interval * (10 - self.cur_stage - 1) * self.stage_count

    # def update(self, stage, time):
    #     # 判断是否改变当前阶段
    #     w_a, w_b = 3, 1
    #     if 0 < stage - self.cur_stage < 2 and self.idx - self.last_stage_idx >= self.stage_count:
    #         self.cur_stage = stage
    #
    #         self.stage_count = self.idx - (self.last_stage_idx + 1)
    #
    #         self.last_stage_idx = self.idx - 1
    #
    #     self.idx += 1
    #     if self.last_time == 0:
    #         self.last_time = time
    #     self.interval = time - self.last_time
    #     self.last_time = time
    #     if not self.stage_count == 0:
    #         self.end_time = time + \
    #                         (self.stage_count - (self.idx - (self.last_stage_idx + 1))) * self.interval + \
    #                         self.interval * (10 - self.cur_stage - 1) * self.stage_count

    def result(self, time):
        rest_time = (self.end_time - time) / 60
        probability = time / self.end_time * 100
        return rest_time, probability
