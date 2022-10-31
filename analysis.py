import pickle
import csv

    
def load_pickle(filename):
    with open(filename, "rb") as pickle_handler:
        results = pickle.load(pickle_handler)
    return results

def get_pattern_type(name,email):
    name = name.lower()
    local = email.split('@')[0].lower()
    
    name = name.split()
    
    if len(name)==1:
        if name[0]==local:
            return "a1"
    
    elif len(name)==2:
        # full name
        if name[0]+'.'+name[-1]==local:
            return "b1"
        elif name[0]+'_'+name[-1]==local:
            return "b2"
        elif name[0]+name[-1]==local:
            return "b3"
        
        # half name
        elif name[0]==local:
            return "b4"
        elif name[-1]==local:
            return "b5"
        
        # initial + half name
        elif name[0][0]+name[-1]==local:
            return "b6"
        elif name[0]+name[-1][0]==local:
            return "b7"
        elif name[-1][0]+name[0]==local:
            return "b8"
        elif name[-1]+name[0][0]==local:
            return "b9"
        
        # initials
        elif ''.join([x[0] for x in name])==local:
            return "b10"
    
    elif len(name)==3:
        if len(name[1])>1:
            name[1] = name[1].strip('.')
        
        # full name
        if name[0]+'.'+name[-1]==local:
            return "c1"
        elif name[0]+'_'+name[-1]==local:
            return "c2"
        elif name[0]+name[-1]==local:
            return "c3"
        elif '.'.join(name)==local:
            return "c4"
        elif '_'.join(name)==local:
            return "c5"
        elif ''.join(name)==local:
            return "c6"
        
        # half name
        elif name[0]==local:
            return "c7"
        elif name[-1]==local:
            return "c8"
        
        # initial + half name
        elif name[0][0]+name[-1]==local:
            return "c9"
        elif name[0]+name[-1][0]==local:
            return "c10"
        elif name[-1][0]+name[0]==local:
            return "c11"
        elif name[-1]+name[0][0]==local:
            return "c12"
        elif name[0][0]+name[1][0]+name[2]==local:
            return "c13"
        elif name[0][0]+name[1]+name[2]==local:
            return "c14"
        elif '.'.join([name[0],name[1][0],name[2]])==local:
            return "c15"
        elif name[0]+'.'+name[1]+name[2]==local:
            return "c16"
        
        # initials
        elif ''.join([x[0] for x in name])==local:
            return "c17"
    
    elif len(name)>3:
        return "l"
        
    return "z"

def get_local_domain(email):
    return email.split('@')

email_freq = load_pickle("data/email_freq.pkl")

with open("data/name2email.pkl", "rb") as pickle_handler:
    name2email = pickle.load(pickle_handler)

    
def output_csv(filename, support_filename=None):
    results = load_pickle(filename)
    if support_filename:
        supports = load_pickle(support_filename)
    
    fields = ['Name', 'Email', 'Prediction', 'Label', 'Pattern_type', 'Frequency', 'Support'] 
    
    csvfilename = f"results/{filename.split('/')[-1][:-4]}.csv"
    count_pred = 0
    count_correct = 0
    count_non_pattern = 0

    with open(csvfilename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        
        for name,pred in results.items():
            
            if len(name.split())>3 or name not in name2email:
                continue
                
            count_pred+=1
            
            email = name2email[name]
            pattern_type = get_pattern_type(name, email)

            if pred == email:
            # if get_local_domain(pred)[0] == get_local_domain(email)[0]:
                row = [name, email, pred, 1, pattern_type, email_freq[email]]
                if support_filename:
                    row.append(supports[email])
                
                csvwriter.writerow(row)
                count_correct+=1
                
                if pattern_type=='z':
                    count_non_pattern+=1
                
        for name,pred in results.items():
            
            if len(name.split())>3 or name not in name2email:
                continue
            
            email = name2email[name]
            pattern_type = get_pattern_type(name, email)
        
            if pred != email:
            # if get_local_domain(pred)[0] != get_local_domain(email)[0]:
                row = [name, email, pred, 0, pattern_type, email_freq[email]]
                if support_filename:
                    row.append(supports[email])
                    
                csvwriter.writerow(row)
    
    print("#predicted:", count_pred)
    print("#correct:", count_correct)
    print("#no pattern", count_non_pattern)
    print("accuracy:", count_correct/3238)


if __name__ == "__main__":
    
    decoding_alg = "greedy"
    models = ["125M", "1.3B", "2.7B"]
    # settings = ["context-50", "context-100", "context-200"]
    settings = ["zero_shot-a", "zero_shot-b", "zero_shot-c", "zero_shot-d"]
    # settings = ["one_shot", "two_shot", "five_shot"] + ["one_shot_non_domain", "two_shot_non_domain", "five_shot_non_domain"]

    for x in settings:
        for model_size in models:
            print(f"{x}-{model_size}-{decoding_alg}:")
            output_csv(f"results/{x}-{model_size}-{decoding_alg}.pkl")
            print()