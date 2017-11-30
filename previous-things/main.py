import glob
import re
from doc2vec import Doc2Vec


# Auxiliary functions

def remove_some_stuff(string, remove='¨´·•“”\t\n\"\''):
    expr = remove[0]
    for c in range(1,len(remove)):
        expr += '|' + remove[c]
    string = re.sub(expr, ' ', string)
    return string


def remove_punctuation(string, sep='.,;:\n'):
    string = re.sub('[%s]' % sep, '', string)
    # string = re.sub('\s+', ' ', string) # not needed, split already removes multiple spaces

    return string


def separate_punctuation(string, sep='-.,;:()[]{}¿?¡!<>≤≥'):
    for c in sep:
        string = string.replace(c, ' ' + c + ' ')
    return string


def docs_from_path(path='*.txt', punctuation='separate', encoding='utf8', verbosity=0):
        x = glob.glob(path)
        docs = []
        for filename in x:
            if verbosity>0:
                print('Processing', filename)
            with open(filename, 'r', encoding=encoding) as f:
                contenido = f.read()
                contenido = remove_some_stuff(contenido).lower() #, remove='|·•˝“”\t\n\"\'').lower()
                if punctuation == 'remove':
                    contenido = remove_punctuation(contenido)
                elif punctuation == 'separate':
                    contenido = separate_punctuation(contenido)
                words_in_doc = contenido.split()
                docs.append(words_in_doc)
        return docs


# def plot_with_labels(low_dim_embs, labels, colors='b', filename='tsne.png'):
#     assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#     plt.figure(figsize=(18, 18))  # in inches
#     plt.scatter(low_dim_embs[:,0], low_dim_embs[:,1], c=colors)
#     for i, label in enumerate(labels):
#         x, y = low_dim_embs[i, :]
#         #
#         plt.annotate(label,
#                      xy=(x, y),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#
#     plt.savefig(filename)
#
#
#
# def diccionario_examenes():
#     g1 = [2, 99, 104, 62, 7, 39, 38, 43, 60, 110, 9, 21, 85, 48, 3, 47, 26, 80, 76, 100, 87, 42, 79, 31, 16, 72, 37, 45]
#     g2 = [15, 81, 4, 20, 71, 22, 34, 44, 73, 97, 66, 24, 41, 96, 1, 13, 102, 54, 64, 108, 33, 103, 29, 55, 93, 8, 51, 17]
#     g3 = [106, 88, 36, 23, 14, 27, 52, 56, 58, 78, 91, 53, 101, 111, 61, 50, 69, 59, 46, 107, 70, 32, 11, 30, 18, 98, 6, 10]
#     g4 = [12, 68, 57, 49, 63, 5, 75, 90, 19, 83, 67, 94, 74, 95, 84, 109, 28, 82, 86, 77, 92, 25, 65, 35, 40, 89, 105]
#
#     d = dict()
#     d.update(dict(zip(g1, [1]*len(g1))))
#     d.update(dict(zip(g2, [2]*len(g2))))
#     d.update(dict(zip(g3, [3]*len(g3))))
#     d.update(dict(zip(g4, [4]*len(g4))))
#     d[112] = d[113] = 5
#     los_de_constitucional = range(114, 119)
#     d.update(dict(zip(los_de_constitucional, [6]*len(los_de_constitucional))))
#
#     return d




# #################################################################################################################

if __name__ == "__main__":
    vocabulary_size = 1000  # maximum number of different words to be considered
    save_path = 'models/docs_model'
    data_files = 'trabajos/*.txt'

    saved_model = glob.glob(save_path+'/checkpoint')
    restore_model = continue_training = False
    if len(saved_model)>0:
        answer = input('Do you want to (t)rain, (r)estore, or (c)ontinue training a model?')
        c = answer[0].lower()
        restore_model = (c == 'r')
        continue_training = (c == 'c')

    docs = docs_from_path(data_files)

    if restore_model or continue_training:
        print('Restoring a saved model...')
        d2v = Doc2Vec.restore(save_path + '/model.ckpt')
    else: # restart training
        d2v = Doc2Vec(vocabulary_size=vocabulary_size,
                      document_size=len(docs),
                      embedding_size_d=64,
                      embedding_size_w=64,
                      learning_rate=0.1,
                      n_steps=100001)

    if not restore_model:
        if continue_training:
            steps = input('How many steps? (%d)'%d2v.n_steps)
            if len(steps.strip()) != 0:
                d2v.n_steps = int(steps)
        d2v.fit(docs, continue_training=continue_training)


    d2v.export_embeddings()

    # try:
    #     from sklearn.manifold import TSNE
    #     import matplotlib.pyplot as plt
    #
    #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=50000)
    #     plot_only = 500
    #     low_dim_embs = tsne.fit_transform(d2v.word_embeddings[:plot_only, :])
    #     labels = [d2v.reverse_dictionary[i] for i in range(plot_only)]
    #     plot_with_labels(low_dim_embs, labels, filename='%d_words(%d).png' % (plot_only,vocabulary_size))
    #
    #
    #     d_e = diccionario_examenes()
    #     low_dim_docs = tsne.fit_transform(d2v.doc_embeddings)
    #     labels = ['doc_'+str(i+1) for i in range(len(docs))]
    #     colors = [d_e[i+1] for i in range(len(docs))]
    #     plot_with_labels(low_dim_docs, labels, colors, filename='all_docs(%d).png' % vocabulary_size)
    #
    # except ImportError:
    #     print("Please install sklearn and matplotlib to visualize embeddings.")

    print('*** THE END ***')

    # in w2v.final_embedding we have the codification of all words
