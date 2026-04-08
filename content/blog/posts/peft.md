---
title: "How to finetune Llama2"
date: "2023-08-29T21:10:00+07:00"
lastmod: "2023-08-30T21:10:00+07:00"
author: ["Jiwen Jiang"]
description: "I summarize the reasons why we should finetune and provide a tutorial for finetuing Llama2. It is continuously updated, so stay tuned!"
summary: "I summarize the reasons why we should finetune and provide a tutorial for finetuing Llama2. It is continuously updated, so stay tuned!"
tags: ["LLM", "Finetune", "Llama 2"]
categories: ["LLM"]
cover:
  image: "/media/blog/peft/featured_hu48a7985e8974d2b4d6ca166a6570488b_123773_720x2500_fit_q75_h2_lanczos.webp"
---

<p>Recently, Meta published its latest LLM, Llama2[], and gained tremendous interest in  open source community. Different studies have been undertaken to evaluate Llama2 and therefore utilize Llama2 to improve their own vertical domain LLM that can acquire both domain capability and general LLM skills using both proprietary and public datasets. Before we dive into the tutorial on finetuning Llama2, we have to consider one question: “Why should we finetune Llama2 or must we?”</p>
<p>Typically we want to finetune a large language model with such following hope or objective: it might perform better if I feed it with more domain knowledge. Well. that’s true but should be with more tricks and computation.</p>
<p>From the course provided by Sharon Zhou[2], we can acquire the reason why we should finetune under some specific circumstances and the common differences between finetuning and prompt. We can conclude the finetuning advantages and disadvantages as follows:</p>
<ul>
<li></li>
</ul>
<h4 id="llama2-info">Llama2 info</h4>
<ul>
<li>Parameter 70B</li>
</ul>
<h4 id="llama2-performance">Llama2 Performance</h4>
<p>As it said Llama2 is almost the same powerful as GPT-3.5 except for coding whereas codeLlama can make up the shortage [].</p>
<h4 id="other-llms-based-on-llama-2">Other LLMs based on Llama 2</h4>
<ul>
<li>Lemur (Pre-training,100B Token with code and text) &amp; Lemur-Chat (Supervised Fine-tuning with 300K examples)</li>
</ul>
<h4 id="computation-estimation">Computation estimation</h4>
<p>With all these above, we perhaps can not hesitate to finetune some open-source LLMs (e.g. Llama 2 and  ChatGLM).</p>
<h5 id="reference">Reference</h5>
<p>[1] <a href="http://xueshu.baidu.com/" target="_blank" rel="noopener"></a></p>
<p>[2] <a href="https://en.wikipedia.org/wiki/Main_Page" target="_blank" rel="noopener"></a></p>
