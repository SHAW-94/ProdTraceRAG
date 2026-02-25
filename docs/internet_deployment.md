# Internet-Facing Deployment Notes

本版本补丁在不破坏本地演示用法的前提下，增加了面向互联网部署所需的最小安全控制：

- Bearer Token 身份认证（公共/管理员双角色）
- IP 允许列表（公共与管理员接口可分开配置）
- `/ingest_local` 本地路径允许列表
- `/config` / `/traces/*` 默认脱敏
- `/chunk` 改为管理员接口
- 可关闭 `/docs` 文档页面

## 重要行为变更

### 1) `/config`
- 现在默认返回**脱敏**配置（不暴露路径、模型名）
- 如需管理员视图，使用：
  - `GET /config?scope=admin`（需管理员 Token）

### 2) `/traces/recent` 和 `/traces/get`
- 现在需要管理员 Token
- 默认返回脱敏数据（query/answer/snippet/path 被遮罩）
- 只有在 `TRACE_ALLOW_RAW_EXPORT=1` 且 `raw=true` 时才返回原始内容

### 3) `/ingest_local`
- 支持原字段 `folder`（兼容）
- 服务端内部字段改为 `folder_path`（向后兼容 alias）
- 可通过 `INGEST_REQUIRE_ALLOWLIST=1` + `INGEST_ALLOWED_ROOTS=/path1,/path2` 启用本地路径白名单

### 4) `/ask` 返回 citations
- 新增 `source_ref` 字段（推荐使用）
- `source` 字段保留兼容，但现在为脱敏/相对路径形式
- `snippet` 可通过 `EXPOSE_CITATION_SNIPPETS=0` 关闭

## 推荐反向代理

- 使用 Nginx / Traefik
- 限制请求体大小和并发
- 开启 HTTPS
- 仅将反向代理暴露到公网
- API 服务监听 `127.0.0.1`

## 最小上线清单

- [ ] 设置 `AUTH_REQUIRED=1`
- [ ] 设置公共与管理员 Token
- [ ] 关闭 docs（`ENABLE_DOCS=0`）
- [ ] 配置 `INGEST_ALLOWED_ROOTS`
- [ ] 配置管理员 IP allowlist
- [ ] 验证 `/config` 与 `/traces/*` 返回为脱敏内容
