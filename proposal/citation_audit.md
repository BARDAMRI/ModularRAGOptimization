# בדיקת ציטוטים – research_proposal_short.pdf

נבדקו כל 16 הפריטים בביבליוגרפיה של הגרסה הקצרה, וכל המקומות בטקסט שבהם הם מצוטטים. נבדקו שלושה דברים: האם המאמר קיים, האם פרטי הביבליוגרפיה נכונים (מול רשומות dblp ו-ACL Anthology), והאם המאמר באמת תומך בטענה שהוא מצוטט עבורה (מול התקציר של המאמר).

**בקצרה:** כל 16 המאמרים קיימים. בשלושה יש שגיאה בפרטים הביבליוגרפיים. בשישה המאמר לא תומך בטענה, או תומך בה רק חלקית. בנוסף, שני מאגרי נתונים מוזכרים בלי ציטוט.

## טבלת סיכום

| # | פריט | קיים | פרטים ביבליוגרפיים | התאמה להקשר | מה לעשות |
|---|------|------|--------------------|-------------|----------|
| 1 | Ethayarajh 2019 | ✓ | ✗ הכותרת מסתיימת ב-**Embeddings**, לא ב-representations | ✓ אנאיזוטרופיה | לתקן כותרת |
| 2 | Gao et al. 2023 (HyDE) | ✓ | ✓ ACL 2023, עמ' 1762–1777 | ✓ | אפשר לחזק את השערה D (ראו למטה) |
| 3 | Hofstätter et al. 2021 (TAS-B) | ✓ | ✓ SIGIR 2021, עמ' 113–122 | ✗ חלשה | להסיר או לנסח מחדש |
| 4 | Karpukhin et al. 2020 (DPR) | ✓ | ✓ EMNLP 2020, עמ' 6769–6781 | ✓ (הערה קטנה) | DPR משתמש במכפלה פנימית, לא בקוסינוס |
| 5 | Lee et al. 2020 (BioBERT) | ✓ | ✓ Bioinformatics 36(4):1234–1240 | ✗ חלקית | לנסח מחדש או להוסיף MedCPT |
| 6 | Lewis et al. 2020 (RAG) | ✓ | ✓ NeurIPS 2020 | ✓ | — |
| 7 | Nentidis et al. 2023 (BioASQ) | ✓ | ✗ **רשימת המחברים שגויה** | חלקית | לתקן מחברים, לנסח מחדש |
| 8 | Reimers & Gurevych 2019 (SBERT) | ✓ | ✓ עמ' 3982–3992 (לפי ACL Anthology) | ✓ | — |
| 9 | Robertson & Zaragoza 2009 | ✓ | ✓ FnTIR 3(4):333–389 | חלקית | להוסיף Rocchio 1971 או Li et al. 2023 |
| 10 | F. Shi et al. 2023 | ✓ | ✓ ICML 2023, עמ' 31210–31227 | ✓ | — |
| 11 | W. Shi et al. 2024 (REPLUG) | ✓ | ✓ NAACL 2024, עמ' 8371–8384 | ✓ | — |
| 12 | Thakur et al. 2021 (BEIR) | ✓ | ✓ NeurIPS D&B 2021 | ✓ | — |
| 13 | Tsatsaronis et al. 2015 | ✓ | ✓ BMC Bioinformatics 16:138 | חלקית | לנסח מחדש |
| 14 | Xiong et al. 2021 (ANCE) | ✓ | ✗ הכותרת היא "...Contrastive **Learning**...", לא "Estimation" | ✓ בעיקרה | לתקן כותרת, לרכך ניסוח |
| 15 | Yu et al. 2021 (ANCE-PRF) | ✓ | ✓ (חסרים עמ' 3592–3596) | ✗ תיאור השיטה שגוי | לנסח מחדש |
| 16 | Zheng et al. 2023 (MT-Bench) | ✓ | ✓ NeurIPS 2023 D&B | ✗ חלשה | להחליף או להוסיף Thomas 2024 ו-Faggioli 2023 |

## פירוט הבעיות

**1. Ethayarajh 2019.** הכותרת המדויקת: *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings*. בגרסה הקצרה המילה האחרונה היא "representations".

**3. Hofstätter 2021 (TAS-B), סעיף 1.2.** הוא מצוטט לטענה ש"hard negatives that lie close to the query but are semantically wrong are a known consequence" של התרחקות השאילתות מהמסמכים. לפי התקציר, המאמר עוסק ביעילות האימון: הוא מקבץ שאילתות לפי נושא בתוך batch ומשתמש ב-distillation משני מורים. אין בו טענה על פער בין שאילתות למסמכים, ואין בו טענה ש-hard negatives נובעים מפער כזה. מומלץ להסיר את הציטוט ולהשאיר את ANCE.

**5. Lee 2020 (BioBERT), סעיף 1.1.** הטענה "Domain pre-training such as BioBERT narrows but does not close this gap" לא נבדקת במאמר. BioBERT נבחן על זיהוי ישויות, חילוץ קשרים ומענה על שאלות, ולא על אחזור צפוף. אפשרויות: לנסח כ-"Domain pre-training such as BioBERT (Lee et al., 2020) improves biomedical language understanding but does not by itself train the encoder to place questions near their answer documents", או להוסיף את MedCPT (Jin et al., 2023), שהוא מאחזר ביו-רפואי.

**7. Nentidis 2023.** המחברים הנכונים לפי dblp ו-LNCS 14163 (עמ' 227–250): Anastasios Nentidis, Georgios Katsimpras, Anastasia Krithara, Salvador Lima-López, Eulàlia Farré-Maduell, Luis Gasco, Martin Krallinger, Georgios Paliouras. השמות "Eirini Vandorou" ו-"Eulalia Merida" אינם מחברי המאמר הזה (Vandorou מופיעה בדוחות BioASQ של שנים קודמות). בנוסף, המשפט שהוא מצוטט בו מתאר את השאלות כ-"short and conversational". ב-BioASQ השאלות נכתבות על ידי מומחים ביו-רפואיים, כך ש-"conversational" לא נתמך (ראו גם סעיף 13).

**9. Robertson & Zaragoza 2009, סעיף 1.3.** סעיף 3.3 שלהם (Blind Feedback) מתאר PRF ואף מציין שהוא "tends to fail on some queries, particularly on queries that are difficult to start with". זה תומך במשפט על החולשה של PRF. אבל אצלם PRF מרחיב את השאילתה במונחים נוספים, ואינו מחליף אותה בצנטרואיד של המסמכים המובילים. לגרסת הצנטרואיד כדאי לצטט את Rocchio (1971), ולגרסה הווקטורית באחזור צפוף את Li et al. (2023, ACM TOIS 41(3), "Pseudo Relevance Feedback with Deep Language Models and Dense Retrievers: Successes and Pitfalls").

**13. Tsatsaronis 2015.** המאמר תומך במבנה של שאלה מול תקצירי PubMed, אבל לא בכך ש"biomedicine is among the hardest settings". לטענה הזו הציטוט המתאים הוא BEIR (Thakur), שכבר מופיע. מומלץ להוריד את "conversational".

**14. Xiong 2021 (ANCE).** הכותרת הנכונה: *Approximate Nearest Neighbor Negative Contrastive **Learning** for Dense Text Retrieval* (ICLR 2021). השם ANCE עצמו הוא ראשי תיבות של "...Contrastive Estimation", ומכאן כנראה הבלבול. מבחינת תוכן, המאמר תומך בכך שמסמכים קרובים לשאילתה במרחב הם לרוב לא רלוונטיים, ולכן משמשים כ-hard negatives. הטענה שזו "תוצאה" של התרחקות שתי האוכלוסיות היא הסקה שלך, וכדאי לנסח אותה בהתאם.

**15. Yu 2021 (ANCE-PRF), סעיף 1.3.** במשפט כתוב ש-PRF "replaces the query with the centroid of top-ranked documents; ANCE-PRF applies this to dense retrieval". לפי התקציר, ANCE-PRF הוא מקודד BERT מאומן "that consumes the query and the top retrieved documents" ומפיק ממנו embedding חדש לשאילתה. אין שם חישוב צנטרואיד. הצעת ניסוח: "...ANCE-PRF instead learns a query encoder that reads the query together with its top-ranked documents (Yu et al., 2021)".

**16. Zheng 2023, סעיף 1.3.** הוא מצוטט לטענה ש-LLMs משמשים כשופטי רלוונטיות ומפיקים ציונים מדורגים. המאמר עוסק בשיפוט של תשובות צ'טבוט (MT-Bench, Chatbot Arena) ולא ברלוונטיות של מסמכים לשאילתה. ציטוטים מתאימים: Thomas et al. 2024 (SIGIR, עמ' 1930–1940) ו-Faggioli et al. 2023 (ICTIR, עמ' 39–50). אפשר להשאיר את Zheng כרקע כללי ל-LLM-as-a-judge.

**2. HyDE, להשערה D.** התקציר של HyDE אומר במפורש שהמסמך ההיפותטי "may contain false details", וטוען שה-dense bottleneck של המקודד מסנן אותם. כדאי לצטט את זה במפורש, כעמדה שהשערה D בוחנת בתחום הביו-רפואי.

## אזכורים ללא ציטוט

- **TREC-COVID** (שלב 1): Roberts et al. 2020, *TREC-COVID: rationale and structure of an information retrieval shared task for COVID-19*, JAMIA 27(9):1431–1436.
- **MS MARCO** (שלב 1): Nguyen et al. 2016, CoCo@NIPS.
- בתקציר, "prior RAG research on this domain provides a basis for comparison": מומלץ לצטט את Xiong et al. 2024 (MedRAG/MIRAGE, Findings of ACL 2024).

## הערות שאינן על ציטוטים (נמצאו תוך כדי)

- **מרחקים לפי ציון (סעיף 3.2):** המספרים "0.89, 0.72, 0.63, 0.54, 0.40 and 0.10" לא תואמים את הנתונים בריפו. הערכים בפועל לציונים 0.1, 0.2, 0.4, 0.6, 0.8 ו-1.0 הם 0.89, 0.84, 0.77, 0.63, 0.40 ו-0.22.
- **"the GT gets the top score in 195 of 196 queries":** בנתונים, ה-GT קיבל 1.0 ב-186 מתוך 187 השאילתות שבהן הוא דורג.
- **"Unfiltered PRF ... drops top-1 ... to 24%":** הרצת ה-PRF בריפו (`analysis/csv/iterative_prf`) בונה את הצנטרואיד רק ממסמכים שעברו סינון של ה-LLM. 24% מתאים לסף ≥0.5, ו-31.1% לסף 1.0. לא מצאתי הרצה לא מסוננת.
- **שער:** כתוב "The Institute of Information Systems Engineering" במקום "המחלקה למערכות מידע". גם התרגום העברי "אחזור מבוססת שליפה" ל-RAG שגוי.

## BibTeX מתוקן

```bibtex
@inproceedings{ethayarajh2019contextual,
  author    = {Ethayarajh, Kawin},
  title     = {How Contextual are Contextualized Word Representations? {C}omparing the Geometry of {BERT}, {ELMo}, and {GPT-2} Embeddings},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)},
  pages     = {55--65},
  year      = {2019}
}

@inproceedings{xiong2021ance,
  author    = {Xiong, Lee and Xiong, Chenyan and Li, Ye and Tang, Kwok-Fung and Liu, Jialin and Bennett, Paul N. and Ahmed, Junaid and Overwijk, Arnold},
  title     = {Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@inproceedings{nentidis2023bioasq11,
  author    = {Nentidis, Anastasios and Katsimpras, Georgios and Krithara, Anastasia and Lima-L{\'o}pez, Salvador and Farr{\'e}-Maduell, Eul{\`a}lia and Gasco, Luis and Krallinger, Martin and Paliouras, Georgios},
  title     = {Overview of {BioASQ} 2023: The Eleventh {BioASQ} Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering},
  booktitle = {Experimental IR Meets Multilinguality, Multimodality, and Interaction (CLEF 2023)},
  series    = {Lecture Notes in Computer Science},
  volume    = {14163},
  pages     = {227--250},
  publisher = {Springer},
  year      = {2023}
}

@inproceedings{yu2021anceprf,
  author    = {Yu, HongChien and Xiong, Chenyan and Callan, Jamie},
  title     = {Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback},
  booktitle = {Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM)},
  pages     = {3592--3596},
  year      = {2021}
}

% ---- suggested additions ----
@incollection{rocchio1971relevance,
  author    = {Rocchio, J. J.},
  title     = {Relevance Feedback in Information Retrieval},
  booktitle = {The {SMART} Retrieval System: Experiments in Automatic Document Processing},
  editor    = {Salton, Gerard},
  publisher = {Prentice-Hall},
  pages     = {313--323},
  year      = {1971}
}

@article{li2023prfdense,
  author  = {Li, Hang and Mourad, Ahmed and Zhuang, Shengyao and Koopman, Bevan and Zuccon, Guido},
  title   = {Pseudo Relevance Feedback with Deep Language Models and Dense Retrievers: Successes and Pitfalls},
  journal = {ACM Transactions on Information Systems},
  volume  = {41},
  number  = {3},
  pages   = {62:1--62:40},
  year    = {2023}
}

@inproceedings{thomas2024llmjudge,
  author    = {Thomas, Paul and Spielman, Seth and Craswell, Nick and Mitra, Bhaskar},
  title     = {Large Language Models can Accurately Predict Searcher Preferences},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {1930--1940},
  year      = {2024}
}

@inproceedings{faggioli2023perspectives,
  author    = {Faggioli, Guglielmo and Dietz, Laura and Clarke, Charles L. A. and Demartini, Gianluca and Hagen, Matthias and Hauff, Claudia and Kando, Noriko and Kanoulas, Evangelos and Potthast, Martin and Stein, Benno and Wachsmuth, Henning},
  title     = {Perspectives on Large Language Models for Relevance Judgment},
  booktitle = {Proceedings of the 2023 ACM SIGIR International Conference on Theory of Information Retrieval (ICTIR)},
  pages     = {39--50},
  year      = {2023}
}

@article{roberts2020treccovid,
  author  = {Roberts, Kirk and Alam, Tasmeer and Bedrick, Steven and Demner-Fushman, Dina and Lo, Kyle and Soboroff, Ian and Voorhees, Ellen M. and Wang, Lucy Lu and Hersh, William R.},
  title   = {{TREC-COVID}: Rationale and Structure of an Information Retrieval Shared Task for {COVID-19}},
  journal = {Journal of the American Medical Informatics Association},
  volume  = {27},
  number  = {9},
  pages   = {1431--1436},
  year    = {2020}
}

@inproceedings{nguyen2016msmarco,
  author    = {Nguyen, Tri and Rosenberg, Mir and Song, Xia and Gao, Jianfeng and Tiwary, Saurabh and Majumder, Rangan and Deng, Li},
  title     = {{MS MARCO}: A Human Generated {MA}chine Reading {CO}mprehension Dataset},
  booktitle = {Proceedings of the Workshop on Cognitive Computation (CoCo@NIPS 2016)},
  year      = {2016}
}

@article{jin2023medcpt,
  author  = {Jin, Qiao and Kim, Won and Chen, Qingyu and Comeau, Donald C. and Yeganova, Lana and Wilbur, W. John and Lu, Zhiyong},
  title   = {{MedCPT}: Contrastive Pre-trained Transformers with Large-scale {PubMed} Search Logs for Zero-shot Biomedical Information Retrieval},
  journal = {Bioinformatics},
  volume  = {39},
  pages   = {btad651},
  year    = {2023}
}
```
