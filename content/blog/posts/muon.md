---
title: "Muon"
date: "2026-04-21T09:10:00+07:00"
lastmod: "2026-04-21T09:10:00+07:00"
author: ["Jiwen Jiang"]
description: "I curate some learning materials for CUDA Programming"
summary: "I curate some learning materials for CUDA Programming"
tags: ["Muon","optimization"]
categories: ["Muon","optimization"]
---


## Muon Introduction





## Muon Split

Muon split means we split the fused matrix into funcional indenpendt ones in muon training setting, because muon optimizer treat the matrix as a whole, ie leveraging the matrix inforamtion that is norm informamtion to update the parameters. Typicallly, in conventionally we spit qkv and mlp fused projecion of up and gate, in which way we can improve the preformance. But what if more finegrrained to the head dim. Can we split the matrix considring head info? In the following, we delve into the experiment of fine-grained split accroding to heads.

We use the moddedgpt github repo as our codebase, where use the standard gpt-2 arch as default. We use 8xH800 as default GPUs.

We first test the some setting in gpt-2 arch for align the baseline record. Naviely, we split all head into one sub-matrix. We have tried the following seeting. We split qkv, mlp, and both.


| Model | Split | Speedup | val loss | train loss | hellaswag
|-------|-------|-------|-------|-------|-------|
| gpt-2 | all | 1.0 | 2.0 | 3.0 | 4.0 |
| gpt-2 | qkv | 1.0 | 2.0 | 3.0 | 4.0 |
| gpt-2 | mlp | 1.0 | 2.0 | 3.0 | 4.0 |
| gpt-2 | both | 1.0 | 2.0 | 3.0 | 4.0 |

