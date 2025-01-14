import json

with open('test.json', 'r') as file:
    true_answers = json.load(file)
file.close()

with open('test_shuffle_bis.txt', 'r') as file:
    datas = file.readlines()
    
def find_sentence(sentence, true_answers):
    for category in true_answers:
        for line in true_answers[category]:
            sentence = sentence.replace("\n", "")
            line = line.replace("\n", "")
            if sentence == line:
                return category
            
with open('y_test_shuffle_for_kaggle.csv', 'w') as file:
    i = 0
    file.write(f"ID,Label\n")
    for data in datas:
        file.write(f"{i},{find_sentence(data, true_answers)}")
        file.write("\n")
        i += 1