import storingAndLoading
from sklearn.metrics.pairwise import cosine_similarity


def generateHierarchy(split_entity_list_fromUI, content_depth_needed, content_capture_needed, time_place_weight,
                      content_weight, topic_interest_keyword, from_date_keyword, to_date_keyword, cluster_method):
    ratio_limit = 95
    ui_parameters = storingAndLoading.load_ui_parameters()

    if cluster_method == "Hubble":
        if ui_parameters["split_entity_list_fromUI"] != split_entity_list_fromUI or ui_parameters[
            "content_capture_needed"] != content_capture_needed or ui_parameters[
            "time_place_weight"] != time_place_weight or ui_parameters["content_weight"] != content_weight or \
                ui_parameters["topic_interest_keyword"] != topic_interest_keyword or ui_parameters[
            "from_date_keyword"] != from_date_keyword or ui_parameters["to_date_keyword"] != to_date_keyword:
            # run_WEHONA(split_entity_list_fromUI, 10, content_capture_needed, time_place_weight, content_weight,
            #            topic_interest_keyword, from_date_keyword, to_date_keyword, ratio_limit)
                print("no method")
        elif ui_parameters["content_depth_needed"] != content_depth_needed:
            alter_WEHONA(content_depth_needed)
    elif cluster_method == "Voyager":
        if ui_parameters["content_depth_needed"] != content_depth_needed:
            alter_Top2Vec(content_depth_needed)

    ui_parameters["split_entity_list_fromUI"] = list(dict.fromkeys(split_entity_list_fromUI))
    ui_parameters["content_depth_needed"] = content_depth_needed
    ui_parameters["content_capture_needed"] = content_capture_needed
    ui_parameters["time_place_weight"] = time_place_weight
    ui_parameters["content_weight"] = content_weight
    ui_parameters["topic_interest_keyword"] = topic_interest_keyword
    ui_parameters["from_date_keyword"] = from_date_keyword
    ui_parameters["to_date_keyword"] = to_date_keyword

    storingAndLoading.store_ui_parameters(ui_parameters)


def search_node(search_term, method_name):
    if method_name == "Hubble":
        vectorizer = storingAndLoading.dynamic_load_tfidf_vectorizer_hubble()
        X_dict = storingAndLoading.dynamic_load_tfidf_array_hubble()
    elif method_name == "Voyager":
        vectorizer = storingAndLoading.dynamic_load_tfidf_vectorizer_voyager()
        X_dict = storingAndLoading.dynamic_load_tfidf_array_voyager()
    search_term_emd = vectorizer.transform([search_term])
    sim_cluster = {}
    for key, term_emd in X_dict.items():
        sim = cosine_similarity(search_term_emd, term_emd)[0][0]
        if sim > 0:
            sim_cluster["".join(key).replace("cluster_", "")] = sim
    if len(sim_cluster) > 0:
        return sim_cluster
    else:
        return "no_cluster"


def alter_WEHONA(content_depth_needed):
    static_news = storingAndLoading.load_static_news()
    static_search = storingAndLoading.static_load_cluster_name_dict_news()
    static_X = storingAndLoading.static_load_tfidf_array_hubble()
    nodes = static_news["nodes"]
    weights_list = static_news["weights_list"]
    content_depth_by_weight = static_news["weights_list"][content_depth_needed - 1]
    nodes_updated = []
    search_updated = {}
    X_updated = {}
    for node in nodes:
        if node["split_with_size"] >= content_depth_by_weight:
            nodes_updated.append(node)
            cluster_number = "cluster_" + str(node["id"])
            search_updated[cluster_number] = static_search[cluster_number]
            X_updated[cluster_number] = static_X[cluster_number]
    static_news["nodes"] = nodes_updated
    storingAndLoading.storeDynamicNews(static_news)
    storingAndLoading.dynamic_store_cluster_name_dict_news(search_updated)
    storingAndLoading.dynamic_store_tfidf_array_hubble(X_updated)


def alter_Top2Vec(content_depth_needed):
    static_top2vec_news = storingAndLoading.load_static_top2vec_news()
    static_top2vec_search = storingAndLoading.static_load_cluster_name_dict_top2vec()
    static_X = storingAndLoading.static_load_tfidf_array_voyager()
    nodes = static_top2vec_news["nodes"]
    nodes_updated = []
    search_updated = {}
    X_updated = {}
    for node in nodes:
        if node["level"] <= content_depth_needed:
            nodes_updated.append(node)
            cluster_number = "cluster_" + str(node["id"])
            search_updated[cluster_number] = static_top2vec_search[cluster_number]
            X_updated[cluster_number] = static_X[cluster_number]
    static_top2vec_news["nodes"] = nodes_updated
    storingAndLoading.storeDynamictop2vecNews(static_top2vec_news)
    storingAndLoading.dynamic_store_cluster_name_dict_top2vec(search_updated)
    storingAndLoading.dynamic_store_tfidf_array_voyager(X_updated)


# def run_nothing():
#     print("hi")


def generate_custer_summary(cluster_method_no):
    cluster_method_no_list = cluster_method_no.split(":")
    cluster_method = cluster_method_no_list[0]
    cluster_no = cluster_method_no_list[1]
    if cluster_method == "Hubble":
        summaries = storingAndLoading.load_summaries_hubble()
        if cluster_no in summaries:
            return summaries[cluster_no]
        else:
            # news = storingAndLoading.load_dynamic_news()
            # cluster_summary = helper.generate_custer_summary(news, cluster_no)
            cluster_summary = "sentence_transformers"
            summaries[cluster_no] = cluster_summary
            storingAndLoading.store_summaries_hubble(summaries)
            return cluster_summary
    elif cluster_method == "Voyager":
        summaries = storingAndLoading.load_summaries_voyager()
        if cluster_no in summaries:
            return summaries[cluster_no]
        else:
            # news = storingAndLoading.load_dynamic_top2vec_news()
            # cluster_summary = helper.generate_custer_summary(news, cluster_no)
            cluster_summary = "sentence_transformers"
            summaries[cluster_no] = cluster_summary
            storingAndLoading.store_summaries_voyager(summaries)
            return cluster_summary


def generate_custer_what(cluster_method_no):
    cluster_method_no_list = cluster_method_no.split(":")
    cluster_method = cluster_method_no_list[0]
    cluster_no = cluster_method_no_list[1]
    if cluster_method == "Hubble":
        whats = storingAndLoading.load_whats_hubble()
        if cluster_no in whats:
            return whats[cluster_no]
        else:
            # news = storingAndLoading.load_dynamic_news()
            # cluster_what = helper.generate_custer_what(news, cluster_no)
            cluster_what = "sentence_transformers"
            whats[cluster_no] = cluster_what
            storingAndLoading.store_whats_hubble(whats)
            return cluster_what
    elif cluster_method == "Voyager":
        whats = storingAndLoading.load_whats_voyager()
        if cluster_no in whats:
            return whats[cluster_no]
        else:
            # news = storingAndLoading.load_dynamic_top2vec_news()
            # cluster_what = helper.generate_custer_what(news, cluster_no)
            cluster_what = "sentence_transformers"
            whats[cluster_no] = cluster_what
            storingAndLoading.store_whats_voyager(whats)
            return cluster_what
