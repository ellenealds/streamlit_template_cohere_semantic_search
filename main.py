import streamlit as st
import cohere  
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from concurrent.futures import ThreadPoolExecutor
import toml

# Load the secret.toml file
with open('secret.toml') as f:
    secrets = toml.load(f)

# Access the API key value
api_key = secrets['API_KEY']

co = cohere.Client(api_key) 

# add title
st.title("Cofinder")
# add a subtitle
st.subheader("A semantic search tool built for the Cohere community")


# Load the search index
search_index = AnnoyIndex(f=4096, metric='angular')
search_index.load('search_index.ann')

# load the csv file called cohere_final.csv
df = pd.read_csv('cohere_text_final.csv')

def search(query, n_results, df, search_index, co):
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                    model="large",
                    truncate="LEFT").embeddings
    
    # Get the nearest neighbors and similarity score for the query and the embeddings, 
    # append it to the dataframe
    nearest_neighbors = search_index.get_nns_by_vector(
        query_embed[0], 
        n_results, 
        include_distances=True)
    # filter the dataframe to include the nearest neighbors using the index
    df = df[df.index.isin(nearest_neighbors[0])]
    df['similarity'] = nearest_neighbors[1]
    df['nearest_neighbors'] = nearest_neighbors[0]
    df = df.sort_values(by='similarity', ascending=False)
    return df



# define a function to generate an answer
def gen_answer(q, para): 
    response = co.generate( 
        model='command-xlarge-20221108', 
        prompt=f'''Paragraph:{para}\n\n
                Answer the question using this paragraph.\n\n
                Question: {q}\nAnswer:''', 
        max_tokens=100, 
        temperature=0.4)
    return response.generations[0].text

def gen_better_answer(ques, ans): 
    response = co.generate( 
        model='command-xlarge-20221108', 
        prompt=f'''Answers:{ans}\n\n
                Question: {ques}\n\n
                Generate a new answer that uses the best answers 
                and makes reference to the question.''', 
        max_tokens=100, 
        temperature=0.4)
    return response.generations[0].text

def display(query, results):
    # 1. Run co.generate functions to generate answers

    # for each row in the dataframe, generate an answer concurrently
    with ThreadPoolExecutor(max_workers=1) as executor:
        results['answer'] = list(executor.map(gen_answer, 
                                              [query]*len(results), 
                                              results['text']))
    answers = results['answer'].tolist()
    # run the function to generate a better answer
    answ = gen_better_answer(query, answers)
    
    # 2. Code to display the resuls in a user-friendly format

    st.subheader(query)
    st.write(answ)
    # add a spacer
    st.write('')
    st.write('')
    st.subheader("Relevant documents")
    # display the results
    for i, row in results.iterrows():
        # display the 'Category' outlined
        st.markdown(f'**{row["Type"]}**')
        st.markdown(f'**{row["Category"]}**')
        st.markdown(f'{row["title"]}')
        # display the url as a hyperlink
        # add a button to open the url in a new tab
        st.markdown(f'[{row["link"]}]({row["link"]})')
        st.write(row['answer'])
        # collapse the text
        with st.expander('Read more'):
            st.write(row['text'])
        st.write('')

# add the if statements to run the search function when the user clicks the buttons

query = st.text_input('Ask a question about Cohere')
# write some examples to help the user

st.markdown('''Try some of these examples: 
- What is the Cohere API?
- What are embeddings?
- What is the Cohere playground?
- How can I build a chatbot?''')

if st.button('Search'):
    results = search(query, 3, df, search_index, co)
    display(query, results)