import os
import sqlite3 as sql
from csv import writer

import nltk
import requests
import wikipedia
from easygui import *
from flask import Flask, flash, render_template, request, session, redirect, url_for
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")
from collections import defaultdict

import requests as req

from bs4 import BeautifulSoup as BS
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

app = Flask(__name__)


@app.route("/main", methods=["GET", "POST"])
def handle_refresh():
    if "logged_in" in session and session["logged_in"] == False:
        return render_template("index.html")
    # Code to execute when the page is refreshed goes here
    # return 'Page refreshed!'
    return render_template("search.html")


@app.route("/")
def home():
    session["logged_in"] = False
    return render_template("index.html")


@app.route("/gohome")
def homepage():
    return render_template("index.html")


@app.route("/enternew")
def new_user():
    return render_template("signup.html")


@app.route("/addrec", methods=["POST", "GET"])
def addrec():
    if request.method == "POST":
        try:
            nm = request.form["Name"]
            phonno = request.form["MobileNumber"]
            email = request.form["email"]
            unm = request.form["Username"]
            passwd = request.form["password"]

            with sql.connect("multisearch1.db") as con:
                cur = con.cursor()
                cur.execute(
                    "INSERT INTO muser(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",
                    (nm, phonno, email, unm, passwd),
                )
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("index.html")


@app.route("/userlogin")
def login_user():
    return render_template("index.html")


@app.route("/nonews")
def no_news():
    return render_template("error.html")


# @app.route('/predict')
# def info_user():
#     return render_template('info.html')


@app.route("/logindetails", methods=["POST", "GET"])
def logindetails():
    # if "logged_in" in session and session["logged_in"] == False:
    #     return redirect(url_for("login_user"))
    if request.method == "POST":
        usrname = request.form["username"]
        passwd = request.form["password"]

        with sql.connect("multisearch1.db") as con:
            cur = con.cursor()
            cur.execute(
                "SELECT username,password FROM muser where username=? ", (usrname,)
            )
            account = cur.fetchall()

            for row in account:
                database_user = row[0]
                database_password = row[1]
                if database_user == usrname and database_password == passwd:
                    session["logged_in"] = True
                    return render_template("search.html")
                else:
                    # flash("Invalid user credentials")
                    # return render_template("login.html")
                    return None

                    # flash("Invalid user credentials")
                    # session["logged_in"] = False
                    # # return render_template("login.html")
                    # return None


@app.route("/reply", methods=["POST", "GET"])
def user_reply():
    if request.method == "POST":
        ques = request.form["searchword"]

        lowercase = ques.lower()
        print(lowercase)

        # ======================query pre-processing=================================
        import nltk
        from nltk.stem import WordNetLemmatizer

        nltk.download("averaged_perceptron_tagger")
        from nltk.corpus import wordnet

        lemmatizer = WordNetLemmatizer()

        # Define function to lemmatize each word with its POS tag

        # POS_TAGGER_FUNCTION : TYPE 1
        def pos_tagger(nltk_tag):
            if nltk_tag.startswith("J"):
                return wordnet.ADJ
            elif nltk_tag.startswith("V"):
                return wordnet.VERB
            elif nltk_tag.startswith("N"):
                return wordnet.NOUN
            elif nltk_tag.startswith("R"):
                return wordnet.ADV
            else:
                return None

        # sentence = 'karnataka elections'
        sentence = lowercase

        # tokenize the sentence and find the POS tag for each token
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))

        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
        # print(wordnet_tagged)

        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        lemmatized_sentence = " ".join(lemmatized_sentence)

        print(lemmatized_sentence)

        import nltk
        from nltk.corpus import stopwords

        # nltk.download('stopwords')
        # nltk.download('punkt')

        stop_words = set(stopwords.words("english"))

        # sentence = "The quick brown foxes jumped over the lazy dog"
        # sentence = "Introduction to stack"
        tokens = nltk.word_tokenize(
            lemmatized_sentence
        )  # tokenize the sentence into words
        filtered_tokens = [
            token for token in tokens if not token.lower() in stop_words
        ]  # filter out stop words

        filtered_sentence = " ".join(
            filtered_tokens
        )  # join the filtered tokens into a sentence

        print("Filtered sentence:", filtered_sentence)

        # ============================ end of pre processing=================================

        # elif lowercase == 'prime minister of India':
        #     lowercase = 'Prime_Minister_of_India'

        # --------Main Code-----------
        # Specify the title of the Wikipedia page
        # wiki = wikipedia.page(lowercase)

        with open("key.txt", "w") as f:
            f.write(lowercase.title())

        # try:
        #     wiki = wikipedia.page(lowercase)
        # except:
        #     return render_template("error.html")

        f = open("text_files/aSummary.txt", "w")
        f.close()

        f = open("text_files/rankedNews.txt", "w")
        f.close()

        try:
            keywords = wikipedia.search(filtered_sentence)
            # print(keywords)

            # wiki = wikipedia.page(keywords[0])

            try:
                wiki = wikipedia.page(keywords[0])
            except wikipedia.exceptions.DisambiguationError as e:
                print(
                    f"The search term '{key}' is ambiguous. Please select from the following options:"
                )
                for option in e.options:
                    print("- " + option)
            # warnings.filterwarnings('ignore')
            print(
                "****************",
                keywords[0],
                "**********************************************",
            )

            # Extract the plain text content of the page, excluding images, tables, and other data.
            text = wiki.content

            import re

            # Clean text
            text = re.sub(r"==.*?==+", "", text)
            text = text.replace("\n", "")
            # print(text)

            with open("wiki_data.txt", "w", errors="ignore", encoding="utf-8") as f:
                f.write(text)

            # ----------------    Text Summarize ------------

            # Load  text data
            with open("wiki_data.txt", "r", errors="ignore") as f:
                text = f.read()

            # Tokenize the text into sentences
            sentences = sent_tokenize(text)

            # Tokenize each sentence into words
            words = [word_tokenize(sentence) for sentence in sentences]


            # Remove stop words and stem the words
            with open("stopwords.txt", "r", errors="ignore") as f:
                stop_words = f.read()
            # stop_words = set(stopwords.words('kannada'))
            ps = PorterStemmer()
            filtered_words = []
            for sentence in words:
                filtered_sentence = []
                for word in sentence:
                    if word.casefold() not in stop_words:
                        filtered_sentence.append(ps.stem(word))
                filtered_words.append(filtered_sentence)
            # Calculate word frequency
            word_freq = defaultdict(int)
            for sentence in filtered_words:
                for word in sentence:
                    word_freq[word] += 1
            # Calculate sentence score
            sentence_scores = defaultdict(int)
            for i, sentence in enumerate(filtered_words):
                for word in sentence:
                    sentence_scores[i] += word_freq[word]
            # Select top n sentences with highest score
            n = 5
            top_sentences = sorted(
                sentence_scores, key=sentence_scores.get, reverse=True
            )[:n]
            # print("***************Scores***************")
            # print(top_sentences)
            # print("************************************************")

            # print(top_sentences)
            summary = ""
            for i in sorted(top_sentences):
                summary += " "+sentences[i] + " "
            # print(summary)

            # with open("wiki_data.txt","w") as f:
            #     f.write(summary)

            with open(
                "text_files/aSummary.txt", "w", errors="ignore", encoding="utf-8"
            ) as f:
                f.write(content(keywords[0]))

            # print(content(keywords[0]))

            with open(
                "text_files/aSummary.txt", "a", errors="ignore", encoding="utf-8"
            ) as f:
                f.write(summary)
                # f.write("\n\nNews\n\n")

        except:
            # pass
            with open("text_files/aSummary.txt", "w", errors="ignore") as f:
                f.write("No Data! Please try with a more specific query.")

        # # --------------------Main--------------

        # url = "https://timesofindia.indiatimes.com/auto"
        # # url = "https://timesofindia.indiatimes.com/" + lowercase
        # # print(url+"*********************************************************")

        # f = open("business.txt", "a")
        # f.write("\n\n\t\t\t\tNews\n\n")
        # f.close()

        # webpage = req.get(url)
        # trav = BS(webpage.content, "html.parser")
        # M = 1
        # for link in trav.find_all("a"):
        #     # PASTE THE CLASS TYPE THAT WE GET
        #     # FROM THE ABOVE CODE IN THIS AND
        #     # SET THE LIMIT GREATER THAN 35
        #     if (
        #         str(type(link.string)) == "<class 'bs4.element.NavigableString'>"
        #         and len(link.string) > 35
        #     ):
        #         print(str(M) + ".", link.string)
        #         M += 1
        #         with open("business.txt", "a", errors="ignore") as f:
        #             f.write("-->" + link.string)
        #             f.write("\n")
        #             f.close()

        # # ---------------2nd------------------
        # # url = "https://timesofindia.indiatimes.com/"

        # url = "https://timesofindia.indiatimes.com/etimes"

        # webpage = req.get(url)
        # trav = BS(webpage.content, "html.parser")
        # M = 1
        # for link in trav.find_all("a"):
        #     # PASTE THE CLASS TYPE THAT WE GET
        #     # FROM THE ABOVE CODE IN THIS AND
        #     # SET THE LIMIT GREATER THAN 35
        #     if (
        #         str(type(link.string)) == "<class 'bs4.element.NavigableString'>"
        #         and len(link.string) > 35
        #     ):
        #         print(str(M) + ".", link.string)
        #         M += 1
        #         with open("news1.txt", "a", errors="ignore") as f:
        #             f.write("-->" + link.string)
        #             f.write("\n")
        #             f.close()

        try:
            # ==========================gNews=====================
            import requests

            # Replace YOUR_API_KEY with your actual API key
            # api_key = "605a88752b94688868dfea4cb79c101a"
            api_key = "98d4889b60b853ce8566bd9b56e48b37"

            # Set the API endpoint and parameters
            url = "https://gnews.io/api/v4/search"
            # params = {"q": lowercase, "lang": "en", "token": api_key, "max": 10}
            params = {"q": lowercase, "lang": "en", "token": api_key, "max": 10}

            # Make the API request and parse the JSON response
            response = requests.get(url, params=params)
            data = response.json()
            # print(data)

            f = open("text_files/news.txt", "w", errors="ignore")
            f.close()

            f = open("text_files/news.txt", "a", errors="ignore")
            # f.write(str(data))

            for i in range(len(data["articles"])):
                # print()
                f.write(data["articles"][i]["title"])
                f.write(". ")
                f.write(data["articles"][i]["description"])
                f.write("\n")

            f.close()

            # ********************Newscatcher***************************
            # !!!!!!!!!!!!!!!!!!! ask before uncommenting !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # Replace YOUR_API_KEY with your actual API key
            # api_key = "WIMaygpxjoBL9lkzvuGvCcOhEMbuNuimpsYKfNkbHBs"
            # api_key = "iBl-Iuj_q2kwyAsFysDv8yqCMuqKUe48FnsdyFyIM-0"

            # # Set the API endpoint and parameters
            # url = "https://api.newscatcherapi.com/v2/search"
            # params = {
            #     "q": lowercase,
            #     "lang": "en",
            #     "sort_by": "relevancy",
            #     "page": 1,
            #     "page_size": 10,
            # }

            # # params = {
            # #     "q": filtered_sentence,
            # #     "lang": "en",
            # #     "sort_by": "relevancy",
            # #     "page": 1,
            # #     "page_size": 10,
            # # }
            # headers = {"x-api-key": api_key}

            # # Make the API request and parse the JSON response
            # response = requests.get(url, params=params, headers=headers)
            # data = response.json()
            # print(data)

            # f = open("text_files/news.txt", "a", errors="ignore")

            # for i in range(len(data["articles"])):
            #     # print()
            #     f.write(data["articles"][i]["excerpt"])
            #     f.write("\n")

            # f.close()

            # *******************************newsApi**************************

            # # Replace YOUR_API_KEY with your actual API key
            # api_key = "d8f149d7cc8d4d5ba144729cff667092"

            # # Set the API endpoint and parameters
            # url = "https://newsapi.org/v2/everything"
            # params = {
            #     'q': filtered_sentence,
            #     # 'lang': 'en',
            #     # 'country' : 'in',
            #     "apiKey": api_key,
            #     'max': 10
            # }

            # # params = {
            # #     'q': filtered_sentence,
            # #     # 'lang': 'en',
            # #     # 'country' : 'in',
            # #     "apiKey": api_key,
            # #     'max': 10
            # # }

            # # Make the API request and parse the JSON response
            # response = requests.get(url, params=params)
            # data = response.json()
            # print(data)

            # f = open("text_files/news.txt", "w", errors="ignore")
            # f.close()

            # f = open("text_files/news.txt", "a", errors="ignore")
            # # f.write(str(data))

            # for i in range(len(data["articles"]), max):
            #     f.write('\n')
            #     f.write(data["articles"][i]["title"])
            #     f.write(". ")
            #     f.write(str(data["articles"][i]["description"]))
            #     f.write("\n")

            # f.close()

            # ***************remove duplicates************
            import nltk
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # Define the list of sentences
            # sentences = ['This is a sentence.', 'This is also a sentence.', 'This is a similar sentence.', 'This is a different sentence.']
            f = open("text_files/news.txt", "r")
            sentences = f.readlines()
            # print(newsList)
            f.close()

            # Tokenize the sentences
            sentences = [
                sent.lower() for sent in sentences
            ]  # make all sentences lowercase
            tokens = [nltk.word_tokenize(sent) for sent in sentences]
            sentences = [" ".join(token) for token in tokens]

            # Convert the sentences into vectors
            vectorizer = TfidfVectorizer()
            sentence_vectors = vectorizer.fit_transform(sentences)

            # Compute the cosine similarity matrix
            cosine_similarities = cosine_similarity(sentence_vectors)

            # Iterate through the matrix and remove duplicates based on threshold cosine similarity
            duplicate_indices = []
            for i in range(len(cosine_similarities)):
                for j in range(i + 1, len(cosine_similarities)):
                    if cosine_similarities[i][j] > 0.8:
                        duplicate_indices.append(j)

            # Remove duplicate sentences
            unique_sentences = [
                sentences[i]
                for i in range(len(sentences))
                if i not in duplicate_indices
            ]

            # removing all content
            f = open("text_files/newsNoDup.txt", "w")
            f.close()

            # print(unique_sentences)
            with open("text_files/newsNoDup.txt", "a") as f:
                for line in unique_sentences:
                    f.write(line)
                    f.write("\n")

            # ***************ranking tf-idf******************
            import math

            # Define the documents
            docs = unique_sentences

            # Split the documents into words
            words_list = list(
                set([word for doc in docs for word in doc.lower().split()])
            )

            # Calculate the idf for each word in the list
            idf = {}
            for word in words_list:
                doc_count = sum([1 for doc in docs if word in doc.lower()])
                idf[word] = math.log(len(docs) / doc_count)

            # Calculate the tf-idf score for each word in each document
            tfidf_scores = []
            for doc in docs:
                tf_scores = {}
                for word in doc.lower().split():
                    if word not in tf_scores:
                        tf_scores[word] = 0
                    tf_scores[word] += 1
                tfidf_scores.append(
                    {word: tf_scores[word] * idf[word] for word in tf_scores}
                )

            # Calculate the document scores
            doc_scores = []
            for i in range(len(docs)):
                score = sum(tfidf_scores[i].values())
                doc_scores.append((i, score))

            # Sort the documents by score
            doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)

            f = open("text_files/rankedNews.txt", "w")
            f.close()

            # write to file
            with open("text_files/rankedNews.txt", "a", encoding="utf-8") as f:
                for i, score in doc_scores:
                    f.write("\n")
                    f.write(docs[i].title())
                    f.write("\n")

            # with open("text_files/rankedNews.txt", "r") as f:
            # print("============================list========================")
            # print(f.readlines())

            import os

            os.remove("text_files/news.txt")
            os.remove("text_files/newsNoDup.txt")

        except:
            # pass
            with open("text_files/rankedNews.txt", "w") as f:
                f.write("No News! Please try with another query.")

        # Print the sorted documents
        # for i, score in doc_scores:
        #     print(docs[i])

        # ===================end--------------------
        # f = open("news1.txt", "r")
        # # print(f.read())
        # # f1 = open("business.txt", "r")
        # # print(f.read())
        # import numpy as np
        # from sklearn.feature_extraction.text import TfidfVectorizer

        # # Input documents
        # docs = f
        # # f.close()

        # print("tfidf*********************************************************")

        # # Initialize TfidfVectorizer
        # tfidf_vectorizer = TfidfVectorizer()

        # # Fit and transform documents to TF-IDF feature matrix
        # tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

        # # Get feature names
        # feature_names = tfidf_vectorizer.get_feature_names_out()

        # # Print TF-IDF scores for each document and feature
        # for doc_idx, doc in enumerate(docs):
        #     print(f"Document {doc_idx+1}:")
        #     for feature_idx in np.argsort(tfidf_matrix[doc_idx].toarray().flatten())[
        #         ::-1
        #     ]:
        #         feature_name = feature_names[feature_idx]
        #         score = tfidf_matrix[doc_idx, feature_idx]
        #         if score > 0:
        #             print(f"\t{feature_name}: {score:.4f}")

        # f.close()

        # N = 4
        # with open("news1.txt", "r") as filedata:
        #     # Read the file lines using readlines()
        #     linesList = filedata.readlines()
        #     # filedata.close()
        #     print("The following are the first", N, "lines of a text file:")

        #     # Traverse in the list of lines to retrieve the first N lines of a file
        #     for textline in linesList[:N]:
        #         # Printing the first N lines of the file line by line.
        #         print(textline, end="")
        #         with open("text_files/sum2.txt", "a") as f:
        #             f.write(textline)
        #             f.close()

        # N = 8
        # with open("business.txt", "r") as filedata:
        #     # Read the file lines using readlines()
        #     linesList = filedata.readlines()
        #     # filedata.close()
        # print("The following are the first", N, "lines of a text file:")

        # # Traverse in the list of lines to retrieve the first N lines of a file
        # for textline in linesList[:N]:
        #     # Printing the first N lines of the file line by line.
        #     print(textline, end="")
        #     with open("text_files/sum1.txt", "a") as f:
        #         f.write(textline)
        #         f.close()

        # import fileinput
        # import glob

        # file_list = glob.glob("text_files/*.txt")

        # f = open("result_123.txt", "w")
        # f.close()

        # with open("result_123.txt", "a") as f:
        #     for fileText in file_list:
        #         with open(fileText, "r") as file:
        #             f.write(file.read())

        # # with open("result_123.txt", "w") as file:
        # #     input_lines = fileinput.input(file_list)
        # #     file.writelines(input_lines)
        # #     file.write("\n")

        # title = "Final_Output"
        # with open("result_123.txt") as f:
        #     lines = f.read()
        #     # message for our window
        #     # button message by default it is "OK"
        #     button = "Close"

        #     # creating a message box
        #     msgbox(lines, title, button)

        # import os

        # f.close()
        # # os.remove("text_files/sum1.txt")
        # # os.remove("text_files/sum2.txt")
        # os.remove("text_files/aSummary.txt")
        # os.remove("result_123.txt")
        # os.remove("wiki_data.txt")
        # # os.remove("news1.txt")
        # # os.remove("business.txt")
        # # os.remove("etimes.txt")

        # import os

        # response = "updates"

        # return render_template('search.html')
        # return render_template("result.html")

        import webbrowser

        url = "http://127.0.0.1:5501/templates/result.html"  # Replace with the URL of your HTML page
        webbrowser.open(url)
        return redirect(url_for("handle_refresh"))
        # return render_template('resultpred.html', prediction=response)


def content(key):
    return wikipedia.summary(key, sentences=2)


def getNews(cat):
    api_key = "d8f149d7cc8d4d5ba144729cff667092"

    # Set the API endpoint and parameters
    url = "https://newsapi.org/v2/top-headlines"
    if cat == None:
        params = {
            # 'q': 'chennai',
            # 'lang': 'en',
            "country": "in",
            "apiKey": api_key,
            # 'max': 10
        }

        with open("key.txt", "w") as f:
            f.write("General News")
    else:
        params = {
            # 'q': 'chennai',
            # 'lang': 'en',
            "country": "in",
            "category": cat,
            "apiKey": api_key,
            # 'max': 10
        }

        with open("key.txt", "w") as f:
            f.write(cat.title())

    # Make the API request and parse the JSON response
    response = requests.get(url, params=params)
    data = response.json()
    # print(data)

    f = open("text_files/newsApi.txt", "w", errors="ignore")
    f.close()

    f = open("text_files/newsApi.txt", "a", errors="ignore", encoding="utf-8")
    # f.write(str(data))

    for i in range(len(data["articles"])):
        # print()
        f.write("\n")
        f.write(data["articles"][i]["title"])
        f.write(". ")
        f.write(str(data["articles"][i]["description"]))
        f.write("\n")

    f.close()

    # =========================remove duplicates==================================


    # ================================== tf-idf ranking ==============================
    import math

    # Define the documents
    f = open("text_files/newsApi.txt", "r", encoding="utf-8")
    docs = f.readlines()
    f.close()

    # Split the documents into words
    words_list = list(
        set([word for doc in docs for word in doc.lower().split()])
    )

    # Calculate the idf for each word in the list
    idf = {}
    for word in words_list:
        doc_count = sum([1 for doc in docs if word in doc.lower()])
        idf[word] = math.log(len(docs) / doc_count)

    # Calculate the tf-idf score for each word in each document
    tfidf_scores = []
    for doc in docs:
        tf_scores = {}
        for word in doc.lower().split():
            if word not in tf_scores:
                tf_scores[word] = 0
            tf_scores[word] += 1
        tfidf_scores.append(
            {word: tf_scores[word] * idf[word] for word in tf_scores}
        )

    # Calculate the document scores
    doc_scores = []
    for i in range(len(docs)):
        score = sum(tfidf_scores[i].values())
        doc_scores.append((i, score))

    # Sort the documents by score
    doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)

    f = open("text_files/newsApi.txt", "w", encoding="utf-8")
    f.close()

    # write to file
    with open("text_files/newsApi.txt", "a", encoding="utf-8") as f:
        for i, score in doc_scores:
            f.write("\n")
            f.write(docs[i].title())
            f.write("\n")
            

    with open("text_files/newsApi.txt", "r", encoding="utf-8") as f:
        content = f.read()
        # print(content.strip())
        with open("text_files/newsApi.txt", "w", encoding="utf-8") as f1:
            f1.write(content.strip())

    # with open("text_files/rankedNews.txt", "r") as f:
    # print("============================list========================")
    # print(f.readlines())

    # import os

    # os.remove("text_files/newsApi.txt")
    # os.remove("text_files/newsNoDup.txt")


    # ====================output==============================

    import webbrowser

    url = "http://127.0.0.1:5501/templates/navNews.html"  # Replace with the URL of your HTML page
    webbrowser.open(url)

    # title = "News"
    # with open("text_files/newsApi.txt") as f:
    #     lines = f.read()

    # os.remove("text_files/newsApi.txt")

    # # message for our window

    # # button message by default it is "OK"
    # button = "Close"

    # # creating a message box
    # msgbox(lines, title, button)
    # M = "News Generated"

    # # return render_template("search.html", prediction=M)
    return redirect(url_for("handle_refresh"))


# =================================== general news===============================
@app.route("/predictinfo")
def predictin():
    return getNews(None)


# =======================business news============================================
@app.route("/predictinfo1")
def predictin1():
    return getNews("business")


# =================================== entertainment ===============================
@app.route("/predictinfo2")
def predictin2():
    return getNews("entertainment")


# =================================== sports ===============================
@app.route("/predictinfo3")
def predictin3():
    return getNews("sports")


# =================================== technology  ===============================
@app.route("/predictinfo4")
def predictin4():
    return getNews("technology")


@app.route("/logout")
def logout():
    session["logged_in"] = False
    # print(session["logged_in"])
    return render_template("index.html")


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
