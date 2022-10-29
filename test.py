from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

import pandas as pd   # pip install pandas

df = pd.read_excel(r'D:\download\1.xls', sheet_name='Лист1')
print(df)

def compare(desc, cat):
    # Two lists of sentences
    sentences = desc.split('.')

    category = [cat] * len(sentences)

    #Compute embedding for both lists
    embeddings1 = model.encode(sentences, convert_to_tensor=True)
    embeddings2 = model.encode(category, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    #Output the pairs with their score
    #for i in range(len(sentences)):
    #    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], category[i], cosine_scores[i][i]))
    for i in cosine_scores:
        if (i >= .5):
            return True
    return False
    

desc = 'Международная компания «TgT Oilfield ServiceS» занимается промыслово-геофизическими исследованиями, мониторингом месторождений углеводородов и геолого-гидродинамическим моделированием. Компания объединяет несколько проектов, которые являются резидентами Фонда «Сколково» (ООО «МИКС», ООО «Сонограм», ООО «Термосим»), все разработки и производство оборудования проходят на территории России.ООО «Сонограм» разработала спектральный шумомер «SNL Locator» для регистрации акустических данных в широком ди'
category = 'Синтез образовательного контента'
compare(desc, category)

