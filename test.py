from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Two lists of sentences
desc = 'Международная компания «TgT Oilfield ServiceS» занимается промыслово-геофизическими исследованиями, мониторингом месторождений углеводородов и геолого-гидродинамическим моделированием. Компания объединяет несколько проектов, которые являются резидентами Фонда «Сколково» (ООО «МИКС», ООО «Сонограм», ООО «Термосим»), все разработки и производство оборудования проходят на территории России.ООО «Сонограм» разработала спектральный шумомер «SNL Locator» для регистрации акустических данных в широком ди'
sentences = desc.split('.')

category = ['Синтез образовательного контента'] * len(sentences)

#Compute embedding for both lists
embeddings1 = model.encode(sentences, convert_to_tensor=True)
embeddings2 = model.encode(category, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], category[i], cosine_scores[i][i]))

