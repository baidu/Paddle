
<p align="center">
<img align="center" src="doc/imgs/logo.png", width=1600>
<p>

--------------------------------------------------------------------------------

[English](./README.md) | 简体中文 | [日本語](./README_ja.md)

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

欢迎来到 PaddlePaddle GitHub

飞桨(PaddlePaddle)以百度多年的深度学习技术研究和业务应用为基础，是中国首个自主研发、功能完备、 开源开放的产业级深度学习平台，集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体。目前，飞桨累计开发者1070万，服务企业23.5万家，基于飞桨开源深度学习平台产生了86万个模型。飞桨助力开发者快速实现AI想法，快速上线AI业务。帮助越来越多的行业完成AI赋能，实现产业智能化升级。

## 安装

### PaddlePaddle 最新版本: [3.0-rc](https://github.com/PaddlePaddle/Paddle/tree/release/3.0-rc)

跟进 PaddlePaddle 最新特性请参考我们的[版本说明](https://github.com/PaddlePaddle/Paddle/releases)

### 安装最新稳定版本

``` sh
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

更多安装信息详见官网 [安装说明](https://www.paddlepaddle.org.cn/install/quick)

PaddlePaddle用户可领取**免费Tesla V100在线算力资源**，训练模型更高效。**每日登陆即送8小时**，[前往使用免费算力](https://aistudio.baidu.com/aistudio/index)。

## 四大领先技术

- **开发便捷的产业级深度学习框架**

    飞桨深度学习框架采用基于编程逻辑的组网范式，对于普通开发者而言更容易上手，符合他们的开发习惯。同时支持声明式和命令式编程，兼具开发的灵活性和高性能。网络结构自动设计，模型效果超越人类专家。

- **支持超大规模深度学习模型的训练**

    飞桨突破了超大规模深度学习模型训练技术，实现了支持千亿特征、万亿参数、数百节点的开源大规模训练平台，攻克了超大规模深度学习模型的在线学习难题，实现了万亿规模参数模型的实时更新。
    [查看详情](https://github.com/PaddlePaddle/Fleet)

- **支持多端多平台的高性能推理部署工具**

    飞桨不仅广泛兼容第三方开源框架训练的模型部署，并且为不同的场景的生产环境提供了完备的推理引擎，包括适用于高性能服务器及云端推理的原生推理库 [Paddle Inference](https://www.paddlepaddle.org.cn/inference/master/guides/introduction/index_intro.html)，全场景、易用灵活、极致高效的AI推理部署工具，支持云边端部署工具 [FastDeploy](https://github.com/PaddlePaddle/FastDeploy)，针对于移动端、物联网场景的轻量化推理引擎 [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)，以及在浏览器、小程序等环境下使用的前端推理引擎 [Paddle.js](https://www.paddlepaddle.org.cn/paddle/paddlejs)。同时，透过与不同场景下的主流硬件高度适配优化及异构计算的支持, 飞桨的推理性能也领先绝大部分的主流实现。

- **面向产业应用，开源开放覆盖多领域的工业级模型库。**

    飞桨官方支持100多个经过产业实践长期打磨的主流模型，其中包括在国际竞赛中夺得冠军的模型；同时开源开放200多个预训练模型，助力快速的产业应用。
    [查看详情](https://github.com/PaddlePaddle/models)

## 文档

我们提供 [英文](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html) 和 [中文](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html) 文档

- [使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)：或许你想从深度学习基础开始学习飞桨

- [应用实践](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/index_cn.html)：使用飞桨搭建你的模型，更高效的完成深度学习任务

- [API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)：新的 API 支持代码更少更简洁的程序

- [贡献方式](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/08_contribution/index_cn.html)：参与飞桨社区开源贡献的指南

## 开源社区

- [Github Issues](https://github.com/PaddlePaddle/Paddle/issues)：提交安装/使用问题、报告bug、建议新特性、沟通开发计划等
- 社区活动：

  - 入门：快乐开源活动（[热身打卡 + 常规赛](https://github.com/PaddlePaddle/Paddle/issues/56689)、[启航计划](https://github.com/PaddlePaddle/Paddle/issues/58497)）
  - 进阶：飞桨黑客马拉松（[开源贡献个人挑战赛](https://github.com/PaddlePaddle/Paddle/issues/57262)、[大模型应用与创意赛](https://github.com/PaddlePaddle/Paddle/issues/57585)、[飞桨护航计划集训营](https://github.com/PaddlePaddle/Paddle/issues/57264)）

- 社区组织：
  - 技术交流组织：[飞桨核心框架贡献者俱乐部 PFCC](https://github.com/PaddlePaddle/community/tree/master/pfcc)
  - 社区治理组织：[飞桨社区开源发展工作组 PPOSDWG](https://github.com/PaddlePaddle/community/tree/master/pposdwg)

- 社区博客：<https://pfcc.blog/>

## 课程

- [服务器部署](https://aistudio.baidu.com/aistudio/course/introduce/19084)：详细介绍高性能服务器端部署实操，包含本地端及服务化Serving部署等
- [端侧部署](https://aistudio.baidu.com/aistudio/course/introduce/22690)：详细介绍端侧多场景部署实操，从移动端设备、IoT、网页到小程序部署

## 版权和许可证

PaddlePaddle由[Apache-2.0 license](LICENSE)提供
