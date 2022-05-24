import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pytextrank

vectorizer = TfidfVectorizer()

nlp = spacy.load('en_core_web_md')
nlp.add_pipe("textrank")

model_name = 'paraphrase-MiniLM-L3-v2'
model = SentenceTransformer(model_name)


def generate_custer_summary(news, cluster_no):
    text_dict = news["text_dict"]
    doc_ids_in_cluster = news["cluster_dict"][cluster_no]
    total_text_from_cluster = ""
    summary_sentence = "<strong>->  </strong>"
    summary_sentence_list = []
    summary_sentence_dict = {}
    docs_required = 10
    split = int(len(doc_ids_in_cluster) / docs_required)
    elements_needed = []
    counter = -1
    for i in doc_ids_in_cluster:
        counter = counter + 1
        if counter == 0:
            elements_needed.append(i)
        if counter == split:
            counter = -1
    for doc_id in elements_needed:
        total_text_from_cluster = total_text_from_cluster + text_dict[str(doc_id)] + ". "
    summaryDoc = nlp(total_text_from_cluster.strip(". "))
    for summary_sent_doc in summaryDoc._.textrank.summary(limit_phrases=100, limit_sentences=50):
        summary_sentence_list.append(summary_sent_doc)

    for index, summary_sent in enumerate(summary_sentence_list):
        if len(summary_sentence_dict) > 4:
            break
        if index == 0:
            summary_sentence_dict[str(summary_sent)] = model.encode(str(summary_sent))
        else:
            flag_list = []
            for summary_sent_key, summary_sent_value in summary_sentence_dict.items():
                doc_emb = model.encode(str(summary_sent))
                sim = round(cosine_similarity([doc_emb], [summary_sent_value]).item(), 3)
                if sim < 0.7:
                    flag_list.append("add")
                else:
                    flag_list.append("delete")
            if not "delete" in flag_list:
                summary_sentence_dict[str(summary_sent)] = doc_emb
    for key in summary_sentence_dict.keys():
        summary_sentence = summary_sentence + key + ". " + "<br>" + "<strong>->  </strong>"
    return summary_sentence.rstrip("<strong>->  </strong>")


def generate_custer_what(news, cluster_no):
    docs_dict = news["text_dict"]
    doc_ids_in_cluster = news["cluster_dict"][cluster_no]
    total_text_from_cluster = ""
    what_sentence = ""
    what_sentence_list = []
    what_sentence_dict = {}
    docs_required = 10
    split = int(len(doc_ids_in_cluster) / docs_required)
    elements_needed = []
    counter = -1
    for i in doc_ids_in_cluster:
        counter = counter + 1
        if counter == 0:
            elements_needed.append(i)
        if counter == split:
            counter = -1
    for doc_id in elements_needed:
        total_text_from_cluster = total_text_from_cluster + docs_dict[str(doc_id)] + ". "
    whatDoc = nlp(total_text_from_cluster.strip(". "))
    for phrase in whatDoc._.phrases:
        what_sentence_list.append(phrase.text)

    for index, what_sent in enumerate(what_sentence_list):
        if len(what_sentence_dict) > 3:
            break
        if index == 0:
            what_sentence_dict[str(what_sent)] = model.encode(str(what_sent))
        else:
            flag_list = []
            for what_sent_key, what_sent_value in what_sentence_dict.items():
                doc_emb = model.encode(str(what_sent))
                sim = round(cosine_similarity([doc_emb], [what_sent_value]).item(), 3)
                if sim < 0.3:
                    flag_list.append("add")
                else:
                    flag_list.append("delete")
            if not "delete" in flag_list:
                what_sentence_dict[str(what_sent)] = doc_emb

    for key in what_sentence_dict.keys():
        what_sentence = what_sentence + key + ", "

    return what_sentence.strip(", ")
