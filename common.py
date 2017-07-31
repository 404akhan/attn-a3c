#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import random
import time
import threading
import multiprocessing
import numpy as np
from tqdm import tqdm
from six.moves import queue
import matplotlib.pyplot as plt 
from skimage import color

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.utils.stats import *

### my code
from model_attn import *
from collections import deque, namedtuple
### end

def f(player, func, attn_net_play, verbose=False):
    def g(s):
        spc = player.get_action_space()
        act = func([[s]])[0][0].argmax()
        if random.random() < 0.001:
            act = spc.sample()
        if verbose:
            print(act)
                
        ### my code        
        state_arr = [np.expand_dims(color.rgb2gray(s[:, :, 3*i:3*(i+1)]), axis=2) for i in range(4)]
        state = np.concatenate(state_arr, axis=2)

        # state | 84 x 84 x 4
        # act | int
        f.replay_memory.append(f.Transition(state, act))
        if len(f.replay_memory) > f.upd_init_size: # train
            f.counter += 1
            samples = random.sample(f.replay_memory, f.batch_size)
            states_batch, action_batch = map(np.array, zip(*samples))
            acc = f.attn_net.train_(states_batch, action_batch)
            f.accuracy_arr.append(acc)

            if f.counter % 100 == 0:
                print('update %d, accuracy %.2f%%' % (f.counter, np.mean(f.accuracy_arr)))
                f.accuracy_arr = []
                if f.counter % 5000 == 0: f.attn_net.save_model(f.counter)

        if attn_net_play: # play
            act = f.attn_net.action_(state)
        ### end
        return act
    return np.mean(player.play_one_episode(g))

### my code
f.counter = 0
f.accuracy_arr = []

f.Transition = namedtuple("Transition", ["state", "action"])
replay_memory_size = 100 * 1000
f.replay_memory = deque([], replay_memory_size)
f.upd_init_size = 10 * 1000
f.batch_size = 16
f.attn_net = Attn(num_heads=4, batch_size=f.batch_size, lr=1e-3)
if f.attn_net.cuda_exist:
    f.attn_net.cuda()
### end

def play_model(cfg, player, game_name):
    f.attn_net.set_game_name(game_name)
    predfunc = OfflinePredictor(cfg)
    counter_games = 0
    
    while True:   
        counter_games += 1
        attn_net_play = counter_games%10 == 0

        score = f(player, predfunc, attn_net_play)
        print(counter_games, attn_net_play, score)

def eval_with_funcs(predictors, nr_eval, get_player_fn):
    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(train=False)
                while not self.stopped():
                    try:
                        score = f(player, self.func)
                        # print("Score, ", score)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()
    try:
        for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
            r = q.get()
            stat.feed(r)
        logger.info("Waiting for all the workers to finish the last run...")
        for k in threads:
            k.stop()
        for k in threads:
            k.join()
        while q.qsize():
            r = q.get()
            stat.feed(r)
    except:
        logger.exception("Eval")
    finally:
        if stat.count > 0:
            return (stat.average, stat.max)
        return (0, 0)


def eval_model_multithread(cfg, nr_eval, get_player_fn):
    func = OfflinePredictor(cfg)
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    mean, max = eval_with_funcs([func] * NR_PROC, nr_eval, get_player_fn)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))


class Evaluator(Triggerable):
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean, max = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('mean_score', mean)
        self.trainer.monitors.put_scalar('max_score', max)


def play_n_episodes(player, predfunc, nr):
    logger.info("Start evaluation: ")
    for k in range(nr):
        if k != 0:
            player.restart_episode()
        score = f(player, predfunc)
        print("{}/{}, score={}".format(k, nr, score))
