import json

with open('y_pred_shuffle.txt', 'r') as file:
    datas = file.readlines()

# with open('y_test_shuffle_for_kaggle.txt', 'w') as file:
#     i = 0
#     file.write(f"ID,Usage,Label\n")
#     for data in datas:
#         file.write(f"{i},Private,{data}") if i % 2 == 1 else file.write(f"{i},Public,{data}")
#         i += 1

# with open('y_test_shuffle.txt', 'r') as file:
#     datas = file.readlines()

with open('y_test_shuffle_for_kaggle.csv', 'w') as file:
    i = 0
    file.write(f"ID,Label\n")
    for data in datas:
        file.write(f"{i},{data}")
        i += 1
