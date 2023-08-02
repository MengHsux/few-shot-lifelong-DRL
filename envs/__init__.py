#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym.envs.registration import register

register(
    'AntNavi-v1',
    entry_point='envs.mujoco.ant:AntNaviEnv',
    max_episode_steps=1000
)
