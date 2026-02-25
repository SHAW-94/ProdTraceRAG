# 开源发布补丁清单（按仓库结构）

> 这份清单与补丁文件配套，目标是：**最小修改**地把当前仓库升级为可开源发布、并支持更安全的互联网部署（认证/允许列表/默认脱敏/最小权限）。

## 一、核心代码变更（已在补丁中）

### `app/api.py`
1. **新增安全依赖接入**
   - 引入 `app.security`，统一处理：
     - Bearer Token 鉴权
     - 公共/管理员最小权限控制
     - IP 允许列表
     - `/config` 和 `/traces` 默认脱敏
     - `/ingest_local` 路径允许列表
2. **接口权限**
   - `POST /ask` -> 公共权限（`require_public`）
   - `POST /set_config` -> 管理员权限（`require_admin`）
   - `POST /reset_index` -> 管理员权限
   - `POST /ingest_local` -> 管理员权限
   - `GET /traces/recent` -> 管理员权限
   - `GET /traces/get` -> 管理员权限
   - `GET /chunk` -> 新增管理员权限接口（给评测脚本使用）
3. **默认脱敏**
   - `/config` 默认返回脱敏视图（路径、模型名隐藏）
   - `GET /config?scope=admin` 返回管理员视图（需管理员 Token）
   - trace 写入与读取默认脱敏（query/answer/snippet/path）
4. **本地摄取安全**
   - `/ingest_local` 增加路径校验与允许列表支持：
     - `INGEST_REQUIRE_ALLOWLIST`
     - `INGEST_ALLOWED_ROOTS`
   - `glob_pattern` 增加允许列表校验（避免任意 glob）
5. **字段兼容与规范**
   - `IngestLocalReq` 内部字段名改为 `folder_path`，仍兼容旧请求字段 `folder`（alias）
   - `Citation` 新增 `source_ref`（推荐使用）
   - `Citation.source` 保留兼容，但输出为脱敏/相对路径
6. **输入约束**
   - `AskReq` 增加 `q/topk/min_score` 参数边界校验
   - `read_text_file` 增加单文件最大读取字符数限制（可配）

### `app/security.py`（新增）
新增安全控制模块，封装：
- `make_fastapi_app()`
- `require_public() / require_admin()`
- `build_config_response()`
- `sanitize_trace_on_write() / sanitize_trace_item()`
- `normalize_and_validate_ingest_folder()`
- `path_ref()` 等路径脱敏工具

### `app/llm_gen.py`
- 增加错误信息脱敏（API key / Bearer token 等不会直接出现在异常文本里）

### `eval/run_eval.py`
- 增加 Bearer Token 请求头支持（环境变量读取）
- `/set_config`、`/config`、`/chunk` 使用管理员 Token
- `/ask` 继续使用公共 Token

### `scripts/verify_features.py`
- 增加 Bearer Token 请求头支持
- 管理员接口改为管理员 Token 调用
- `/config` 使用 `?scope=admin` 便于验证完整配置

### `scripts/make_sample_report.py`
- 增加 Bearer Token 请求头支持
- 管理员接口改为管理员 Token 调用
- 兼容 `source_ref` 字段
- 示例 curl 文本增加认证头示例

---

## 二、开源标准文件（已在补丁中新增）

- `LICENSE`（MIT）
- `SECURITY.md`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `.env.example`
- `.github/workflows/opensource-sanity.yml`
- `docs/internet_deployment.md`

---

## 三、README / README_CN / .gitignore（建议补充，执行命令里已给一键追加脚本）

由于原始 README 内容可能与你本地版本不同，补丁文件不直接覆盖 README；执行命令会自动追加以下内容（若文件存在）：
- README / README_CN：添加“互联网部署安全说明”链接
- `.gitignore`：追加 `.env`, `.chroma/`, `logs/`, `__pycache__/`, `*.pyc` 等常见忽略项（去重追加）

---

## 四、环境变量（上线建议）

### 必需（公网部署）
- `AUTH_REQUIRED=1`
- `PRODTRACERAG_API_TOKEN`
- `PRODTRACERAG_ADMIN_TOKEN`

### 强烈建议
- `INGEST_REQUIRE_ALLOWLIST=1`
- `INGEST_ALLOWED_ROOTS=/srv/prodtracerag/corpus`
- `ENABLE_DOCS=0`
- `TRACE_REDACT_ON_WRITE=1`
- `TRACE_ALLOW_RAW_EXPORT=0`
- `PUBLIC_CONFIG_SHOW_PATHS=0`
- `PUBLIC_CONFIG_SHOW_MODEL=0`

### 可选增强
- `PUBLIC_ALLOWLIST_CIDRS`
- `ADMIN_ALLOWLIST_CIDRS`
- `TRUST_PROXY_HEADERS=1`（仅在可信反向代理后）

---

## 五、接口兼容性说明（重点）

### `POST /ingest_local`
旧请求（仍可用）：
```json
{"folder": "/path/to/corpus", "glob_pattern": "**/*.md"}
```

新内部字段：
- `folder_path`（通过 alias 兼容旧 `folder`，客户端暂时不用改也能跑）

### `GET /config`
- 默认：`/config` -> 脱敏公共视图
- 管理员：`/config?scope=admin` -> 完整视图（需管理员 Token）

### `GET /traces/*`
- 现在需要管理员 Token
- 默认返回脱敏内容
- 仅当 `TRACE_ALLOW_RAW_EXPORT=1` 且 `raw=true` 时允许导出原始内容

### `/ask` citations
新增字段：
- `source_ref`（推荐客户端改用这个字段）

保留兼容字段：
- `source`（值现在是脱敏/相对路径，不再保证是绝对路径）
