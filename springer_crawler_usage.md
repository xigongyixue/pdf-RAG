# Springer Journal Article Crawler

从 Springer 期刊页面批量下载文章的 PDF 和 BibTeX 文件。

## 用法

```bash
python springer_crawler.py [OPTIONS]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--journal JOURNAL` | `778` | Springer 期刊 ID（如 778 = The VLDB Journal）。从 URL 获取：`link.springer.com/journal/<ID>` |
| `--volume VOLUME` | `34` | 卷号 |
| `--cookies {chrome,firefox,edge,opera,none}` | `chrome` | 加载浏览器 Cookie 以访问付费墙内的 PDF。使用 `none` 跳过 Cookie（仅 Open Access） |
| `--retry-missing` | - | 仅重新下载之前失败的文章 |

### 示例

```bash
# 默认行为：使用 Chrome Cookie 下载 VLDB Journal Vol 34
python springer_crawler.py

# 下载其他期刊/卷
python springer_crawler.py --journal 123 --volume 10

# 不使用 Cookie（仅下载 Open Access 文章）
python springer_crawler.py --cookies none

# 补下载之前失败的文章
python springer_crawler.py --retry-missing
```

## 输出结构

下载内容保存到 `papers/` 根目录下，格式为：

```
papers/
└── {期刊名缩写}{年份}(volume{卷号})/
    ├── pdfs/
    │   ├── MSAD A deep dive into model selection for time series anomaly detection.pdf
    │   └── ...
    ├── bibs/
    │   ├── MSAD A deep dive into model selection for time series anomaly detection.bib
    │   └── ...
    └── articles.json
```

- 期刊名缩写：去除 "The"、"Journal" 等词（如 `The VLDB Journal` → `VLDB`）
- 示例：`VLDB2025(volume34)`、`Nature2024(volume10)`

### 文件说明

- **pdfs/**: PDF 文件，以文章标题命名
- **bibs/**: BibTeX 文件，每篇一个，可直接导入 LaTeX
- **articles.json**: 包含所有文章的元数据（标题、摘要、作者、DOI、BibTeX 键等）

### BibTeX 格式示例

```bib
@article{Sylligardos2025_72,
  author    = {Sylligardos, Emmanouil and Paparrizos, John and Palpanas, Themis and Senellart, Pierre and Boniol, Paul},
  title     = {{MSAD: A deep dive into model selection for time series anomaly detection}},
  journal   = {The VLDB Journal},
  volume    = {34},
  number    = {72},
  year      = {2025},
  doi       = {10.1007/s00778-025-00949-1},
}
```

## 注意事项

- 默认从 Chrome 读取 Cookie，需要管理员权限
- Cookie 仅在内存中使用，不会写入磁盘
- 已下载的 PDF 不会重复下载
- 每次请求间隔 2 秒，避免触发反爬
- 期刊名称自动从网页标题识别

## 依赖

```bash
pip install requests beautifulsoup4 browser-cookie3
```