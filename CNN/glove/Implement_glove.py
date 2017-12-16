from data_helpers import *
from embedding_loader import *
import sys


loaded_data = load_data_and_labels_another()
test_data = load_test_data()
main("data")

#for line in loaded_data[0]:
   

dictionary1 = {}
dictionary1['x'] = loaded_data[0]
dictionary1['y'] = loaded_data[1]

f = open("data/data_pickling", 'wb')
pickle.dump(dictionary1,f)
f.close()


g = open("data/test_data_pickling", 'wb')
pickle.dump(load_test_data(), g)
g.close()

word_id_convert("data")
test_data_id_convert("data")

