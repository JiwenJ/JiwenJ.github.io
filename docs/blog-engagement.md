# 文章阅读量与匿名点赞

这组改动沿用现有 Hugo/PaperMod 布局，在文章元信息中显示阅读量与赞数，在文末放置点赞区。浅色、深色与手机布局使用同一组组件。文章列表只读取统计，不增加文章浏览量。

## 当前状态

未配置服务端地址时，文章阅读量复用站点已经加载的不蒜子脚本。点赞区和列表统计隐藏；不会显示虚假的零，也不会向占位地址发送请求。

配置 Waline 服务端后，文章阅读量、列表统计和点赞统一读取服务端计数。接入 Waline 后的阅读量独立累计，不会自动继承不蒜子数据。

## 接入

1. 根据 [Waline 官方部署指南](https://waline.js.org/guide/get-started/) 部署服务端与持久数据库。匿名点赞使用文章计数接口，不需要显示评论区。
2. 将允许的站点域名配置为 `jiwenj.github.io`，并设置服务端 `SITE_URL`、`SECURE_DOMAINS`。参考 [服务端环境变量](https://waline.js.org/reference/server/env.html)。
3. 在 `config.yml` 中填写真实服务地址：

```yaml
params:
  blogEngagement:
    enabled: true
    serverURL: "https://your-waline-server.example"
```

4. 构建并发布 Hugo。数据库凭据只应保存在后端；页面配置只包含公开的服务地址。

接口契约核验自 [`@waline/vercel@1.43.4`](https://github.com/walinejs/waline/commit/c4b0f1ebb9a7970d36493d4c3580f3f43e098fa5)。使用其他版本时，请确认 article API 兼容。

## 计数行为

- 文章路径 `.RelPermalink` 是计数键。变更文章 URL 后需要迁移计数。
- 阅读量是 PV：文章页每次加载后自增一次。列表页仅批量读取数值。
- 点赞与取消点赞等待服务器成功返回后更新数量；请求失败保留原状态并允许重试。
- `localStorage` 只保存当前浏览器是否点过赞。所有访客看到的总数来自共享后端。
- 此匿名接口不保证每个自然人只能点赞一次；换浏览器、清除数据或直接请求接口均可能重复计数。

## 本地预览

预览配置应放在仓库外，例如 `/private/tmp/blog-engagement-preview.yml`：

```yaml
baseURL: "http://127.0.0.1:8787/"
params:
  blogEngagement:
    serverURL: "http://127.0.0.1:8787"
    preview: true
```

```sh
hugo --environment development --config config.yml,/private/tmp/blog-engagement-preview.yml --destination /private/tmp/blog-engagement-preview
python3 scripts/preview_server.py --directory /private/tmp/blog-engagement-preview --port 8787
```

预览服务器的示例数字只保存在内存中，页面显示“界面预览 · 演示数据”。不要将预览配置或示例计数发布到正式页面。
