import pandas as pd
import glob
import librosa
from sklearn.model_selection import train_test_split

test_bird_names = {
              "alder_flycatcher":"aldfly",
              "american_goldfinch": "amegfi",
              "buff-bellied_pipit": "amepip"
              }

train_bird_names = {
              "ashy_prinia":"ashpri",
              "black_bulbul":"blabul",
              "forest_elaenia":"forela",
              "yellow-billed_cuckoo":"yelbilcuck",
}


root_path = "XC_Sounds_Split/"


df_dict = {}
df_dict["filepath"] = []
df_dict["bird_name"] = []
df_dict["ebird_code"] =[]

print("Making train df....")
for bird in train_bird_names.keys():
    folder_path = root_path+bird+"/*"

    for ii,filename in enumerate(glob.glob(folder_path)):


        df_dict["filepath"].append(filename)
        df_dict["bird_name"].append(bird)
        df_dict["ebird_code"].append(train_bird_names[bird])


extra_Train,extra_Test = train_test_split(pd.DataFrame(df_dict),test_size=0.2)

df_dict_t = {}
df_dict_t["filepath"] = []
df_dict_t["bird_name"] = []
df_dict_t["ebird_code"] =[]
for bird in test_bird_names.keys():
    folder_path = root_path+bird+"/*"

    for ii,filename in enumerate(glob.glob(folder_path)):


        df_dict_t["filepath"].append(filename)
        df_dict_t["bird_name"].append(bird)
        df_dict_t["ebird_code"].append(test_bird_names[bird])

core_df = pd.DataFrame(df_dict_t)
total_test = pd.concat([core_df,extra_Test],axis=0)

total_test.reset_index(inplace=True)
'''non_bird_path = "motor_demo/*"
df_dict_non = {}
df_dict_non["filepath"] = []
df_dict_non["label"] = []

for filename in glob.glob(non_bird_path):
    print(filename)
    df_dict_non["filepath"].append(filename)
    df_dict_non["label"].append("NOCALL")
'''
def df_checker(df,SR=32000):
    #df = pd.read_csv(df_path)
    new_df_dict = {}
    new_df_dict["filepath"] = []
    new_df_dict["bird_name"] = []
    new_df_dict["ebird_code"] = []
    for ii,filename in enumerate(df["filepath"]):
        print("Iteration: ",ii+1)
        #audio, _ = librosa.load(filename, sr=SR, mono=True, res_type="kaiser_fast")
        try:

            audio,_ = librosa.load(filename,sr=SR,mono=True,res_type="kaiser_fast")
            new_df_dict["filepath"].append(filename)
            new_df_dict["bird_name"].append(df["bird_name"][ii])
            new_df_dict["ebird_code"].append(df["ebird_code"][ii])

        except Exception as e:
            print(e)
            inp = input("Enter y to break")
            if inp=="y":
                break
            print(new_df_dict)


    return pd.DataFrame(new_df_dict)
#csv_path = "kaggle_1_soln/test_df.csv"
df = df_checker(total_test)
df.to_csv("kaggle_1_soln/checked_df_test.csv")

extra_Train.reset_index(inplace=True)
x_df = df_checker(extra_Train)
x_df.to_csv("kaggle_1_soln/checked_df_train.csv")
#df_test = df_checker(pd.DataFrame(df_dict_t))
#df_test.to_csv("kaggle_1_soln/checked_df_test.csv")
#pd.DataFrame(df_dict).to_csv("kaggle_1_soln/test_df.csv")
#pd.DataFrame(df_dict_non).to_csv("kaggle_1_soln/motor.csv")