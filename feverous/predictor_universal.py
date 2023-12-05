import sys
import baseline.predictor.train_verdict_predictor
import shutil
import baseline.predictor.evaluate_verdict_predictor5
import json
import os

def gen_fake_text(data_path):
    fake_txt_dev=""
    f=open(data_path+"/dev.jsonl","r")
    for line in f:
        u=line.replace('evidence','predicted_evidence')
        obj=json.loads(u)
        listq=[]
        for x in obj['predicted_evidence']:
            listrr=[]
            for rt in x['content']:
                listrr+=[rt]

            listq+=listrr
        obj['predicted_evidence']=listq
        if not "NOT ENOUGH INFO" in line:
            fake_txt_dev+=json.dumps(obj)+'\n'

    f.close()
    f=open(data_path+"/fake_dev_cells.jsonl","w")
    f.write(fake_txt_dev)
    f.close()
    


def exec_predictor_alt_devdb(db_path, data_path, exp_name,db_path_test, dev_train, dev_test, usinggt=True, usingretriever=True):
    if  os.path.exists(data_path+'/models/training_args.bin'):
        os.remove(data_path+'/models/training_args.bin')
    f_d_t=open(data_path+"/dev.jsonl","w")
    f_d_s=open(data_path+"/"+dev_train,"r")
    for line in f_d_s:
        f_d_t.write(line)
    f_d_s.close()
    f_d_t.close()
    
    baseline.predictor.train_verdict_predictor.main2(data_path,db_path,data_path+"/models")
    
    
    f_d_t=open(data_path+"/dev.jsonl","w")
    f_d_s=open(data_path+"/"+dev_test,"r")
    for line in f_d_s:
        f_d_t.write(line)
    f_d_s.close()
    f_d_t.close()
    
    if usinggt:

        
        print('############################################################################################################')
        print('Using gt')
        print('############################################################################################################')

        gen_fake_text(data_path)
        baseline.predictor.evaluate_verdict_predictor5.main2(data_path+"/fake_dev_cells.jsonl",data_path+"/models", db_path_test)

        f=open(data_path+"/fake_dev_cells.verdict.jsonl",'r')

        obj=[]

        for line in f:
            obj+=[json.loads(line)]

        f.close()
        res=''
        accuracy=0
        count_refutes=0
        count_supports=0
        count_nei=0

        count_refutes_pred=0
        count_supports_pred=0
        count_nei_pred=0


        precision_num_sup=0
        precision_denum_sup=0
        precision_num_nei=0
        precision_denum_nei=0
        precision_num_ref=0
        precision_denum_ref=0

        recall_denum_sup=0
        recall_denum_ref=0
        recall_denum_nei=0



        for i in range(len(obj)):
            x=dict()
            if 'claim' not in obj[i]:
                print("No claim")
                continue
            else:
                print(obj[i]['claim'])
            x['claim']=obj[i]['claim']
            predicted_label=obj[i]['predicted_label']
            label=obj[i]['label']
            x['label']=label
            x['predicted_label']=predicted_label
            if(predicted_label==label):
                accuracy+=1
                x['label_match']=True
            else:
                x['label_match']=False

            if label=='SUPPORTS':
                count_supports+=1
                recall_denum_sup+=1


            if predicted_label=='SUPPORTS':
                count_supports_pred+=1
                precision_denum_sup+=1
                if x['label_match']:
                    precision_num_sup+=1

            if label=='REFUTES':
                count_refutes+=1
                recall_denum_ref+=1

            if predicted_label=='REFUTES':
                count_refutes_pred+=1
                precision_denum_ref+=1
                if x['label_match']:
                    precision_num_ref+=1


            if label=='NOT ENOUGH INFO':
                count_nei+=1
                recall_denum_nei+=1

            if predicted_label=='NOT ENOUGH INFO':
                count_nei_pred+=1
                precision_denum_nei+=1
                if x['label_match']:
                    precision_num_nei+=1



            res+=json.dumps(x)+"\n"



        #res+="\n\n\naccuracy: " + str(accuracy/(count_nei+count_refutes+count_supports))+", Supports count: "+str(count_supports)+", Refutes count: "+str(count_refutes)+", NEI count: "+str(count_nei)+", Pred Supports count: "+str(count_supports_pred)+", Pred Refutes count: "+str(count_refutes_pred)+", Pred NEI count: "+str(count_nei_pred)
        #res+="\n"+"precision supports: " + str(precision_num_sup/precision_denum_sup)+", recall support : "+ str(precision_num_sup/recall_denum_sup)
        #res+="\n"+"precision refutes: " + str(precision_num_ref/precision_denum_ref)+", recall refutes : "+ str(precision_num_ref/recall_denum_ref)
        #res+="\n"+"precision nei: " + str(precision_num_nei/precision_denum_nei)+", recall nei : "+ str(precision_num_nei/recall_denum_nei)


        f_comp=open(data_path+"/"+exp_name+'_predictor.jsonl','w')
        f_comp.write(res)
        f_comp.close()
        f_scores=open(data_path+"/"+exp_name+'_predictor_scores.json','w')
        dict_sc=dict()
        dict_sc['accuracy']=accuracy/(count_nei+count_refutes+count_supports) if not count_nei+count_refutes+count_supports==0 else 'NAN'
        dict_sc['count_supports']=count_supports
        dict_sc['count_refutes']=count_refutes
        dict_sc['count_nei']=count_nei
        dict_sc['count_supports_pred']=count_supports_pred
        dict_sc['count_refutes_pred']=count_refutes_pred
        dict_sc['count_nei_pred']=count_nei_pred
        dict_sc['precision_supports']=precision_num_sup/precision_denum_sup if not precision_denum_sup==0 else 'NAN'
        dict_sc['precision_refutes']=precision_num_ref/precision_denum_ref if not precision_denum_ref==0 else 'NAN'
        dict_sc['recall_supports']=precision_num_sup/recall_denum_sup if not recall_denum_sup==0 else 'NAN'
        dict_sc['recall_refutes']=precision_num_ref/recall_denum_ref if not recall_denum_ref==0 else 'NAN'
        f_scores.write(json.dumps(dict_sc))
        f_scores.close()
        if  os.path.exists(data_path+'/models/training_args.bin'):
            os.remove(data_path+'/models/training_args.bin')
        print("Predictor : Results written to " + data_path +"/"+exp_name)
#%%

    if usingretriever:
        print('NO############################################################################################################')
        































def exec_predictor_alt_devdb_quick(db_path, data_path, exp_name,db_path_test, dev_train, dev_test, useSavedModel, usinggt=True, usingretriever=True):
    if  os.path.exists(data_path+'/models/training_args.bin'):
        os.remove(data_path+'/models/training_args.bin')
    f_d_t=open(data_path+"/dev.jsonl","w")
    f_d_s=open(data_path+"/"+dev_train,"r")
    for line in f_d_s:
        f_d_t.write(line)
    f_d_s.close()
    f_d_t.close()
    if not useSavedModel:
        baseline.predictor.train_verdict_predictor.main2(data_path,db_path,data_path+"/models")
    
    
    f_d_t=open(data_path+"/dev.jsonl","w")
    f_d_s=open(data_path+"/"+dev_test,"r")
    for line in f_d_s:
        f_d_t.write(line)
    f_d_s.close()
    f_d_t.close()
    
    if usinggt:

        
        print('############################################################################################################')
        print('Using gt')
        print('############################################################################################################')

        gen_fake_text(data_path)
        baseline.predictor.evaluate_verdict_predictor5.main2(data_path+"/fake_dev_cells.jsonl",data_path+"/models", db_path_test)

        f=open(data_path+"/fake_dev_cells.verdict.jsonl",'r')

        obj=[]

        for line in f:
            obj+=[json.loads(line)]

        f.close()
        res=''
        accuracy=0
        count_refutes=0
        count_supports=0
        count_nei=0

        count_refutes_pred=0
        count_supports_pred=0
        count_nei_pred=0


        precision_num_sup=0
        precision_denum_sup=0
        precision_num_nei=0
        precision_denum_nei=0
        precision_num_ref=0
        precision_denum_ref=0

        recall_denum_sup=0
        recall_denum_ref=0
        recall_denum_nei=0



        for i in range(len(obj)):
            x=dict()
            if 'claim' not in obj[i]:
                print("No claim")
                continue
            else:
                print(obj[i]['claim'])
            x['claim']=obj[i]['claim']
            predicted_label=obj[i]['predicted_label']
            label=obj[i]['label']
            x['label']=label
            x['predicted_label']=predicted_label
            if(predicted_label==label):
                accuracy+=1
                x['label_match']=True
            else:
                x['label_match']=False

            if label=='SUPPORTS':
                count_supports+=1
                recall_denum_sup+=1


            if predicted_label=='SUPPORTS':
                count_supports_pred+=1
                precision_denum_sup+=1
                if x['label_match']:
                    precision_num_sup+=1

            if label=='REFUTES':
                count_refutes+=1
                recall_denum_ref+=1

            if predicted_label=='REFUTES':
                count_refutes_pred+=1
                precision_denum_ref+=1
                if x['label_match']:
                    precision_num_ref+=1


            if label=='NOT ENOUGH INFO':
                count_nei+=1
                recall_denum_nei+=1

            if predicted_label=='NOT ENOUGH INFO':
                count_nei_pred+=1
                precision_denum_nei+=1
                if x['label_match']:
                    precision_num_nei+=1



            res+=json.dumps(x)+"\n"



        #res+="\n\n\naccuracy: " + str(accuracy/(count_nei+count_refutes+count_supports))+", Supports count: "+str(count_supports)+", Refutes count: "+str(count_refutes)+", NEI count: "+str(count_nei)+", Pred Supports count: "+str(count_supports_pred)+", Pred Refutes count: "+str(count_refutes_pred)+", Pred NEI count: "+str(count_nei_pred)
        #res+="\n"+"precision supports: " + str(precision_num_sup/precision_denum_sup)+", recall support : "+ str(precision_num_sup/recall_denum_sup)
        #res+="\n"+"precision refutes: " + str(precision_num_ref/precision_denum_ref)+", recall refutes : "+ str(precision_num_ref/recall_denum_ref)
        #res+="\n"+"precision nei: " + str(precision_num_nei/precision_denum_nei)+", recall nei : "+ str(precision_num_nei/recall_denum_nei)


        f_comp=open(data_path+"/"+exp_name+'_predictor.jsonl','w')
        f_comp.write(res)
        f_comp.close()
        f_scores=open(data_path+"/"+exp_name+'_predictor_scores.json','w')
        dict_sc=dict()
        dict_sc['accuracy']=accuracy/(count_nei+count_refutes+count_supports) if not count_nei+count_refutes+count_supports==0 else 'NAN'
        dict_sc['count_supports']=count_supports
        dict_sc['count_refutes']=count_refutes
        dict_sc['count_nei']=count_nei
        dict_sc['count_supports_pred']=count_supports_pred
        dict_sc['count_refutes_pred']=count_refutes_pred
        dict_sc['count_nei_pred']=count_nei_pred
        dict_sc['precision_supports']=precision_num_sup/precision_denum_sup if not precision_denum_sup==0 else 'NAN'
        dict_sc['precision_refutes']=precision_num_ref/precision_denum_ref if not precision_denum_ref==0 else 'NAN'
        dict_sc['recall_supports']=precision_num_sup/recall_denum_sup if not recall_denum_sup==0 else 'NAN'
        dict_sc['recall_refutes']=precision_num_ref/recall_denum_ref if not recall_denum_ref==0 else 'NAN'
        f_scores.write(json.dumps(dict_sc))
        f_scores.close()
        if  os.path.exists(data_path+'/models/training_args.bin'):
            os.remove(data_path+'/models/training_args.bin')
        print("Predictor : Results written to " + data_path +"/"+exp_name)
#%%

    if usingretriever:
        print('NO############################################################################################################')
        

