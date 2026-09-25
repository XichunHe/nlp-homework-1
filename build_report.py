"""Build a Chinese PDF from measured result files; never invent missing scores."""
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parent
font=os.environ.get("REPORT_FONT", "C:/Windows/Fonts/simsun.ttc")
if Path(font).exists(): pdfmetrics.registerFont(TTFont("CN",font))
else: pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
FONT="CN" if Path(font).exists() else "STSong-Light"
body=ParagraphStyle("body",fontName=FONT,fontSize=10.5,leading=16,spaceAfter=8,wordWrap="CJK")
heading=ParagraphStyle("heading",parent=body,fontSize=15,leading=21,spaceBefore=12,spaceAfter=9)
title=ParagraphStyle("title",parent=body,fontSize=23,leading=32,spaceAfter=16)
small=ParagraphStyle("small",parent=body,fontSize=9,leading=13)
cell=ParagraphStyle("cell",parent=body,fontSize=9,leading=13,spaceAfter=0)
story=[]
def p(text,style=body): story.append(Paragraph(escape(text),style))
def h(text):p(text,heading)
def table(rows,widths):
    t=Table([[Paragraph(escape(str(x)),cell) for x in row] for row in rows],colWidths=widths,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8edf1")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#cbd0d5")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.extend([t,Spacer(1,10)])
def read(name):return json.loads((ROOT/"results"/(name+".json")).read_text(encoding="utf-8"))
names=[("binary","Binary BOW"),("frequency","Word Frequency"),("tfidf","TF-IDF"),("glove","GloVe 100d"),("w2v_ag","Word2Vec AG"),("w2v_nyt","Word2Vec NYT"),("bert","BERT-base-uncased")]
results={k:read(k) for k,_ in names if (ROOT/"results"/(k+".json")).exists()}
audit=read("data_audit"); verification=read("verification")
truncation=read("truncation_audit") if (ROOT/"results/truncation_audit.json").exists() else None
p("文本分类实验报告",title)
p("自然语言处理课程作业一  NYT 新闻分类",heading)
p("姓名：何熹淳    学号：2410721",small)
missing=[label for key,label in names if key not in results]
p("运行状态："+("全部七组实验已有实测结果。" if not missing else "已完成 "+str(len(results))+" / 7 组；"+"、".join(missing)+" 待运行。本报告为阶段结果，尚不能作为完整作业提交。"))
h("一  实验目的与数据")
p("比较词袋表示、平均词向量和预训练 BERT 在同一 NYT 新闻分类任务中的效果。线性模型统一使用 Logistic Regression；BERT 对分类任务进行全参数微调。所有报告指标都由本次运行保存的测试预测计算，缺失实验不填入推测数值。")
table([["数据","规模","用途"],["NYT","11,519 篇，3 类","监督分类、训练集 Word2Vec"],["AG News","90,000 篇","无监督训练 Word2Vec"]],[83,130,282])
rows=[["类别","总计","训练","验证","测试"]]
for label in audit["labels"]:rows.append([label,audit["counts"][label],*[audit["split_counts"][s][label] for s in ["train","validation","test"]]])
rows.append(["合计",audit["n"],9215,1152,1152]);table(rows,[135,90,90,90,90])
p("固定 seed=42，采用分层随机打乱，先划分 80% 训练集与 20% 留出集，再把留出集均分为验证集和测试集。整数取整后为 9,215 / 1,152 / 1,152。保存原始行号和数据 SHA-256，所有实验读取同一份 splits.json。")
p("体育类占 75.00%。在测试集全部预测 sports 的多数类基线 Accuracy=75.00%，Macro-F1=28.57%，因此不能只凭准确率评价模型。数据没有空文本或空标签，但有 72 条标准化重复文本；跨集合重复的影响在第四节单独检查。")
p("任务解释：要求文档写“三种”词袋方法，却只列出 Binary 与 Word Frequency。本实验补充 TF-IDF 作为第三种，属于明确假设，提交前可向教师确认。",small)
story.append(PageBreak())
h("二  方法与训练设置")
p("词袋与词向量统一把英文转小写，使用正则提取英文单词、词内单引号及数字，不去停用词。BERT 直接对原始文本使用其预训练 tokenizer。词表、IDF 以及 NYT Word2Vec 均仅从训练集学习，验证集用于选择参数，测试集只用于最终评价。")
table([["方法","文档表示与设置"],["Binary BOW","每个词出现记为 1，否则为 0。保留训练集完整词表。"],["Word Frequency","每个维度表示该词在文档中的原始出现次数。"],["TF-IDF","tf 为原始词频；idf=log((1+N)/(1+df))+1；再作 L2 归一化。"],["GloVe","glove.6B.100d，使用预训练 100 维词向量。"],["Word2Vec AG / NYT","分别在全部 AG 或仅 NYT 训练集训练。Skip-gram，100 维，window=5，min_count=2，negative=5，10 epochs，单线程。"],["BERT","google-bert/bert-base-uncased；max_length=64；完整 3 epochs；全参数微调。"]],[120,375])
p("词向量文档表示为所有有效词向量的算术平均，重复出现的词按出现次数参与平均。OOV 词忽略，分母为实际命中词的数量；若没有有效词则使用 100 维零向量。每个划分的词覆盖率和零向量文档数保存在结果 JSON。")
p("Logistic Regression 使用 L2 正则、lbfgs 和 max_iter=2000，在 C={0.1,1,10} 中按验证集 Macro-F1 选最优值，相同时优先较小 C。选择后不合并验证集重训。")
p("BERT 代码采用 AdamW，学习率 2e-5，weight_decay=0.01，batch size 默认 16，10% warmup 后线性衰减，梯度范数裁剪为 1。每轮评估验证集，完整训练三轮后载入验证 Macro-F1 最高的一轮，最后评测测试集。")
h("三  评价指标")
p("Accuracy 为正确预测数除以测试样本数。对每个类别 k，F1(k)=2TP(k)/(2TP(k)+FP(k)+FN(k))；Macro-F1 是三个类别 F1 的算术平均，分母为零时对应 F1 设为 0。Macro-F1 对每个类别等权，能反映少数类别表现。")
p("实现未对类别设置不同权重；此处的类别不均衡由分层划分、Macro-F1 和逐类指标揭示。本实验是单个固定种子的比较，不能把小幅差异解释为统计显著优势。")
story.append(PageBreak())
h("四  实测结果与复核")
rows=[["模型","测试 Accuracy","测试 Macro-F1","选定 C / 轮次"]]
for key,label in names:
    r=results.get(key)
    rows.append([label,*([f'{r["test"]["accuracy"]*100:.2f}%',f'{r["test"]["macro_f1"]*100:.2f}%',str(r.get("selected_C",r.get("selected_epoch","-")))] if r else ["待运行","待运行","-"])])
table(rows,[160,112,112,111])
best=max(results,key=lambda k:results[k]["test"]["macro_f1"])
best_label=dict(names)[best]
p(f"当前已完成实验中，{best_label} 的测试 Macro-F1 最高，为 {results[best]['test']['macro_f1']*100:.2f}%。这是本次划分下的描述性比较。未完成项目不能参与排名，也不能据此认定 BERT 优于或弱于传统方法。" if missing else f"本次划分下，{best_label} 的测试 Macro-F1 最高，为 {results[best]['test']['macro_f1']*100:.2f}%。排序仅代表本次单种子实验。")
p("Binary 与词频模型保留了直接区分类别的词汇信息。TF-IDF 减弱高文档频率词的权重，可能使主题词更突出。平均词向量把整篇文档压缩到 100 维，降低存储维度，但丢失词序且可能稀释强分类信号；这些是方法层面的解释，并非经过消融验证的因果结论。")
if "w2v_ag" in results and "w2v_nyt" in results:
    ag=results["w2v_ag"]["test"]["macro_f1"];nyt=results["w2v_nyt"]["test"]["macro_f1"]
    p(f"AG 与 NYT 自训练词向量的测试 Macro-F1 分别为 {ag*100:.2f}% 和 {nyt*100:.2f}%。语料规模与领域匹配同时发生变化，不能只把差异归因于其中一个因素。")
p("BERT 被要求截断到 64 个 token，而传统方法读取全文。比较同时包含表示方式与可见文本长度的差别；即使 BERT 完成，也不应把性能差异完全归因于模型结构。")
if truncation:
    p(f"实际分词审计显示，测试集 {truncation['test_truncated_n']:,} / {truncation['test_n']:,} 篇新闻均超过 64 个 token，原文 token 长度中位数为 {truncation['test_median_tokens']:.1f}，90 分位数为 {truncation['test_p90_tokens']:.1f}。这说明 BERT 的输入覆盖远短于词袋模型，是解释结果时必须考虑的限制。")
h("重复文本敏感性")
p("标准化方式为小写并合并连续空白。训练/验证、训练/测试、验证/测试分别存在 14、10、1 个重叠文本。保留课程要求的原始随机划分作为主结果，再仅从测试评价中排除与训练文本重叠的 10 条，剩余 1,142 条，不重新训练或调参。")
rows=[["模型","原 Macro-F1","排除重叠后 Macro-F1"]]
for key,label in names:
    if key in verification["methods"]:
        v=verification["methods"][key];rows.append([label,f'{v["recomputed"]["macro_f1"]*100:.2f}%',f'{v["without_train_duplicates"]["macro_f1"]*100:.2f}%'])
table(rows,[180,140,175])
h("五  错误分析与可复现性")
p(f"以下展示当前最佳已完成模型 {best_label} 的混淆矩阵。行是真实类别，列是预测类别；类别顺序保持 business、politics、sports。")
table([["真实 / 预测",*audit["labels"]],*[[lab,*row] for lab,row in zip(audit["labels"],results[best]["test"]["confusion_matrix"])]],[180,105,105,105])
rows=[["类别","Precision","Recall","F1"]]
for lab in audit["labels"]:
    v=results[best]["test"]["classification_report"][lab]
    rows.append([lab,*[f'{v[k]*100:.2f}%' for k in ["precision","recall","f1-score"]]])
table(rows,[180,105,105,105])
p("以 TF-IDF 的错误为例：原始行号 995（零起始）讨论伦敦奥运会的经济收益，真实类别 sports，预测 business；行号 5358 讨论参议院学生贷款利率协议，真实类别 politics，预测 business。两篇同时涉及体育或政治事件与经济术语，说明类别边界可能交叉；这属于文本内容观察，不等同于已证明模型的具体决策原因。")
if "bert" in results:
    history=results["bert"]["history"]
    values="、".join(f"{v['validation']['macro_f1']*100:.2f}%" for v in history)
    p(f"BERT 完整训练三轮，各轮验证 Macro-F1 依次为 {values}；按预定规则选第 {results['bert']['selected_epoch']} 轮，在测试集得到 {results['bert']['test']['macro_f1']*100:.2f}%。")
p("verify_results.py 使用 Python 标准库独立计数 TP、FP、FN，复算所有已完成模型的 Accuracy 和 Macro-F1，并断言结果与保存 JSON 一致、测试行号顺序一致、真实标签一致、三集合行号互斥且覆盖全体样本。检查结果保存在 verification.json。")
h("复现与交付")
p("从 README.md 安装依赖；依次执行词袋、Word2Vec、GloVe 和 BERT 命令，再执行 verify_results.py 与 build_report.py。模型结果、测试逐条预测与固定划分均包含在 results 中。预训练大权重通过下载脚本或 Hugging Face 获取，不放入 GitHub 普通文件。")
if missing:p("待完成事项："+"、".join(missing)+" 仍没有实测结果。收到 GPU 连接信息后可运行 BERT；最终报告需要在所有实验结束后重新生成。当前报告没有用占位数值替代缺失指标。")
p("复现限制：不同库版本、CPU/GPU 运算与浮点精度可能产生合理差异。固定单线程 Word2Vec 和随机种子降低随机性，但不承诺跨平台逐位一致。重复敏感性检查仅覆盖完全相同的标准化文本，不排除近重复或语义相似内容。")
h("参考资料")
p("1. 课程提供的《作业一要求》与 nyt.csv、ag.csv。",small)
p("2. Stanford GloVe：https://nlp.stanford.edu/projects/glove/",small)
p("3. BERT：https://huggingface.co/google-bert/bert-base-uncased",small)
p("4. GloVe 下载镜像：huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt。解压文件 SHA-256 见 README.md。",small)

def footer(c,doc):
    c.setFont(FONT,8);c.setFillColor(colors.HexColor("#555555"))
    c.drawString(50,30,"作业一  文本分类实验")
    c.drawRightString(A4[0]-50,30,str(doc.page))
doc=SimpleDocTemplate(str(ROOT/"report.pdf"),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=43,bottomMargin=48,title="作业一 文本分类实验报告",author="")
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(ROOT/"report.pdf")
