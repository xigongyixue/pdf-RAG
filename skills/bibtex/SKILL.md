# 学术论文 BibTeX 引用生成

## 1. 技能概述

本技能用于将用户提供的论文信息（如标题、作者、期刊、年份等）或论文链接/DOI，转化为符合国际标准的 BibTeX 格式文本。生成的 BibTeX 条目需确保字段完整、格式规范、无多余冗余信息，并可直接用于 LaTeX 文档中。

## 2. 触发条件

当用户请求生成论文引用、提供论文元数据并要求输出 BibTeX，或者要求格式化参考文献时，触发此技能。

## 3. 标准工作流

1. **提取信息**：从用户输入中提取论文的元数据（作者、标题、年份、期刊/会议名、卷号、期号、页码、DOI等）。
2. **判断类型**：确定文献类型（如 `@article`, `@inproceedings`, `@book`, `@misc` 等）。
3. **生成 Key**：按照"第一作者姓+年份+标题首个实词首字母"的规则生成引用键（Citation Key），如 `Smith2023Deep`。
4. **格式化输出**：严格按照标准模板输出 BibTeX 条目，注意大小写、特殊字符（如 `&` 需转义为 `\&`）和标点符号。
5. **缺失提示**：若关键字段（如 DOI 或页码）缺失，在代码块后简要提示用户。

## 4. 通用 BibTeX 模板库

根据论文类型选择最合适的模板：

### 4.1 期刊论文

适用于发表在学术期刊上的论文。

```bibtex
@article{AuthorYearKeyword,
  author    = {Last1, First1 and Last2, First2},
  title     = {{Title of the Paper in Title Case}},
  journal   = {Full Journal Name},
  year      = {2023},
  volume    = {12},
  number    = {3},
  pages     = {45--60},
  publisher = {Publisher Name},
  doi       = {10.1000/xyz123}
}
```

### 4.2 会议论文

适用于收录在会议论文集（如 CVPR, ACL, ICML 等）中的论文。

```bibtex
@inproceedings{AuthorYearKeyword,
  author    = {Last1, First1 and Last2, First2},
  title     = {{Title of the Conference Paper}},
  booktitle = {Proceedings of the Conference Name},
  year      = {2023},
  pages     = {100--110},
  publisher = {Publisher Name},
  address   = {Conference Location},
  doi       = {10.1000/xyz123}
}
```

## 5. 格式化规则与注意事项

- **作者格式**：姓氏在前，名字在后，中间用逗号隔开；多位作者之间用 `and` 连接（如 `Smith, John and Doe, Jane`）。
- **标题大小写**：BibTeX 中 title 字段建议使用双大括号 `{{...}}` 包裹，以防止 LaTeX 自动将某些字母转为小写。
- **特殊字符转义**：确保标题中的 `%`, `&`, `$`, `#`, `_` 等特殊字符前加上反斜杠 `\` 进行转义。
- **页码格式**：使用双连字符 `--` 表示范围（如 `1--10`）。
- **字段顺序**：尽量保持 `author, title, journal/booktitle, year, volume, number, pages, doi` 的顺序，便于阅读。

## 6. 输出示例

**用户输入：**

> 帮我把这篇论文生成BibTeX：作者 John Smith, Alice Lee；标题 "Deep Learning for NLP & Vision"；发表在 Journal of Artificial Intelligence Research, 2023, Vol 15, No 2, pp 30-45. DOI: 10.1234/jair.2023.001

**AI 输出：**

```bibtex
@article{Smith2023Deep,
  author    = {Smith, John and Lee, Alice},
  title     = {{Deep Learning for NLP \& Vision}},
  journal   = {Journal of Artificial Intelligence Research},
  year      = {2023},
  volume    = {15},
  number    = {2},
  pages     = {30--45},
  doi       = {10.1234/jair.2023.001}
}
```
