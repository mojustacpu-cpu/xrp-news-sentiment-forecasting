# Editorial change log

Scope: the latest public 2023–2026 XRP/USDT manuscript. The title, section hierarchy, research questions, experimental findings and conclusions are preserved. No model, dataset or saved experimental output was changed.

## Main changes

- Clarified that E0/E1B/E2 are the principal comparisons and introduced E1, E3 and E4 by their distinct objectives.
- Kept the supporting price-target LSTM/BiLSTM/BTC/ARIMAX analyses separate from the principal exact24-hour log-return news experiments.
- Distinguished GPT-derived semantic annotations from presence and recency calculated locally from timestamps.
- Connected the hourly availability proxy, zero-news interpretation, causal trailing decay and validation-only half-life selection.
- Merged short related fragments into developed paragraphs, without removing the indicator derivations, equations or scientific caveats.
- Distinguished validation loss used to select checkpoints from price-RMSE used to rank configurations.
- Added missing table references and connected main-text interpretations to the detailed appendix evidence.
- Framed the Discussion subsections by RQ2/RQ3, RQ3, RQ4 and RQ5; retained the independent RQ1 technical comparison and all study boundaries.
- Condensed the Conclusion from six paragraphs to three, retaining the1.406% validation gain,5.358% later deterioration,76.89%/23.11% attribution and the negative HBAR transfer finding.
- Rephrased Highlights accuracy percentages as RMSE changes and limited attribution wording to its model-specific meaning.

## Exact trace

`EDITORIAL_EDITS.json` contains before/after text, section, reason and type for every change. `MANUSCRIPT_DIFF.patch` is the complete unified LaTeX diff. All20 tabular bodies and all9 figure environments are byte-identical; the bibliography and all33 citation occurrences are preserved. The caption phrase “apples-to-apples” was formalised as “matched”.

## Change index

| No. | Section | Type | Purpose |
|---|---|---|
| 1 | Abstract | editorial | Distinguish the principal comparisons from the additional E1/E3/E4 configurations. |
| 2 | Abstract | editorial | Name the explanation method and distinguish attribution from predictive value. |
| 3 | Abstract | editorial | Replace ambiguous influence language with the bounded model-specific conclusion. |
| 4 | Highlights | editorial | Report RMSE changes rather than call them percentage changes in accuracy. |
| 5 | Highlights | editorial | Avoid suggesting that numerical checks certify exact attributions or broad generalisation. |
| 6 | Introduction | editorial | Explain the purpose and target boundary of supporting models before introducing the news experiment. |
| 7 | Introduction | editorial | Make the roadmap match the existing Related work subsections; interpretability remains in its dedicated results section. |
| 8 | Introduction | editorial | Use descriptive rather than potentially inferential significance language in RQ4. |
| 9 | Related work | editorial | Separate GPT-derived semantic attributes from locally calculated timing features and avoid repeating the full feature inventory. |
| 10 | Related work | editorial | Explain the tested temporal representation without assuming a causal news lifespan. |
| 11 | Methodology | editorial | Replace repeated general motivation with a concise methodological roadmap. |
| 12 | Methodology | editorial | Orient the reader to every experiment label without implying a new recurrent architecture. |
| 13 | Methodology | editorial | Keep the incremental-information rationale without repeating the configuration inventory. |
| 14 | Methodology 3.1 | editorial | Keep the forecasting objective and regression interpretation in one developed paragraph. |
| 15 | Technical inputs | editorial | Group the complete input count with the feature-consistency control. |
| 16 | Technical inputs | editorial | Attach the table introduction to the feature inventory it summarises. |
| 17 | Technical inputs | editorial | Connect the moving-average interpretations before the unchanged EMA equation. |
| 18 | Technical inputs | editorial | Join the volatility notation explanation to its interpretation. |
| 19 | Technical inputs | editorial | Resolve a short concluding fragment while preserving the OBV/MFI distinction. |
| 20 | Technical inputs | editorial | Link the market-context rationale with its alignment requirement. |
| 21 | News methodology | editorial | Explain the two model roles together, avoiding an isolated introductory sentence. |
| 22 | News methodology | editorial | Distinguish the saved annotation schema from the fields actually ingested. |
| 23 | News methodology | editorial | Introduce the news-feature table within the paragraph explaining derived timing variables. |
| 24 | News timing | editorial | Combine the information-cutoff rule with the implemented proxy and rounding convention. |
| 25 | News timing | editorial | Keep the zero-delay assumption and concrete hourly-alignment example together. |
| 26 | News timing | editorial | Explain the distinct meanings of zero sentiment, presence and recency without new imputations. |
| 27 | News timing | editorial | Develop the availability limitation with its explanatory example. |
| 28 | News timing | editorial | State the protection and its remaining evidential limit in one paragraph. |
| 29 | Decay methodology | editorial | Move from the hourly representation limitation to the objective of E2 without an abrupt transition. |
| 30 | Decay methodology | editorial | Attach the weighting-equation introduction to the definition of its inputs. |
| 31 | Decay methodology | editorial | Explain why the second decay feature is supplied alongside weighted sentiment. |
| 32 | Decay methodology | editorial | Introduce and interpret the zero-specificity example before the unchanged table. |
| 33 | Decay methodology | editorial | Connect normalisation behaviour to the validation-only choice of half-life. |
| 34 | Architecture | editorial | Introduce early fusion as the implemented operation rather than as a separate generic stage. |
| 35 | Architecture | editorial | Keep the figure pointer with the shared-design clarification. |
| 36 | Architecture | editorial | Retain the cited recurrent mechanism while placing the historical-only constraint immediately alongside it. |
| 37 | Architecture | editorial | Present the actual recurrent-to-output path as a single implementation paragraph. |
| 38 | Architecture | editorial | Replace a repeated configuration summary with a transition to the experiment protocol. |
| 39 | Experimental design | editorial | Separate configuration ranking from the within-run scaled-return checkpoint criterion. |
| 40 | Results table caption | editorial | Use formal terminology while preserving the observations and statistics. |
| 41 | Technical results | editorial | Explain the reference table, evaluation-period boundary and associated research questions. |
| 42 | News experiment objectives | editorial | Introduce E1/E1B/E2/E3/E4 by the question each comparison tests, without new experiment claims. |
| 43 | News results | editorial | Interpret the plotted trajectories without treating visual fit as advance turning-point evidence. |
| 44 | Later evaluation | editorial | Make the frozen evaluation chronology and complete seed reporting explicit. |
| 45 | Attribution | editorial | Name the verification scope explicitly. |
| 46 | Attribution | editorial | Introduce the appendix table and numerical diagnostic figure from the relevant main-text explanation. |
| 47 | Attribution | editorial | Explicitly introduce and interpret the global attribution table and figure. |
| 48 | Experimental design | editorial | Introduce the chronological-design table and distinguish fitting, selection and frozen assessment. |
| 49 | Technical results | editorial | Explicitly introduce the supporting architecture table before interpreting it. |
| 50 | Discussion 6.1 | editorial | Organise the news-value discussion around the research questions and avoid repeating the introduction. |
| 51 | Discussion 6.2 | editorial | Tie the interpretation of decay directly to RQ3. |
| 52 | Discussion 6.3 | editorial | Frame timing limitations as the evidential boundary of the timing research question. |
| 53 | Discussion 6.4 | editorial | State the attribution-versus-predictive-value distinction at the start of the subsection. |
| 54 | Discussion 6.5 | editorial | Connect the existing limitations to the full set of questions without identifying an untested causal mechanism. |
| 55 | Discussion 6.2 | editorial | Link the supplementary six-hour finding to its retained per-seed table. |
| 56 | Conclusion | editorial | Condense repeated findings into three connected paragraphs while retaining all reported quantitative conclusions, attribution distinctions and HBAR caveats. |
| 57 | Appendix | editorial | Introduce all retained detailed tables and diagnostics, identifying their relation to the main research questions. |
| 58 | Training protocol | editorial | Connect checkpoint selection to its detailed epoch and history evidence. |
| 59 | Preamble | layout | Use compact standard float spacing, permit additional floats and top-align float-only pages instead of stretching gaps. |
| 60 | Appendix boundary | layout | Remove the manual page break; flush preceding floats but allow appendix text to continue naturally. |
| 61 | Highlights author line | layout | Restore readable spaces after author-name commas without changing authorship. |
