import sys
import os

# Get the current working directory
# cwd = os.getcwd()
# os.chdir("")#YOUR FEVEOUR DIRECTORY

# Print the current working directory
print("Current working directory: {0}".format(cwd))
import baseline.retriever.train_cell_evidence_retriever
import baseline.retriever.predict_cells_from_table
import json
import traceback
from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage
import itertools
import shutil

#db =  FeverousDB("/content/drive/MyDrive/Colab Notebooks/feverousdata/filtereddb4.db")


def get_cell_value(txt,db):
    try:
        txt=txt.replace('header_cell','cell')
        page_to_request=str(txt.split('_')[0])
        table_to_request=int(txt.split('_')[2])
        row_to_request=int(txt.split('_')[3])
        col_to_request=int(txt.split('_')[4])
        page_json = db.get_doc_json(page_to_request)
        wiki_page = WikiPage(page_to_request, page_json)

        wiki_tables = wiki_page.get_tables() #return list of all Wiki Tables

        wiki_table_0 = wiki_tables[table_to_request]
        wiki_table_0_rows = wiki_table_0.get_rows() #return list of WikiRows

        cells_row_0 = wiki_table_0_rows[row_to_request].get_row_cells()#return list with WikiCells for row 0
        return str(cells_row_0[col_to_request])
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        return 'ERROR'


def gen_fake_dev(data_path):
    fake_txt_dev=""
    #res2=''
    f=open(data_path+"/dev.jsonl","r")
    for line in f:
        #res2+='UUUUUUUUUUUUUUUUUUUU '  + line+'\n'
        u=line.replace('evidence','predicted_evidence')
        obj=json.loads(u)
        listq=[]
        for x in obj['predicted_evidence']:
            listrr=[]
            for rt in x['content']:
                if not 'table_caption' in rt:
                    if not 'header_cell' in rt:
                        listrr+=[[rt.split('_')[0],'table_'+rt.split('_')[2]]]
                    else:
                        listrr+=[[rt.split('_')[0],'table_'+rt.split('_')[3]]]
                else:
                    print(rt)
            listq+=listrr
        obj['predicted_evidence']=listq
        fake_txt_dev+=json.dumps(obj)+'\n'

    f.close()
    #return [fake_txt_dev,res2]
    return fake_txt_dev


def exec_cell_retriever(db_path, data_path, exp_name):
    db=FeverousDB(db_path)
    baseline.retriever.train_cell_evidence_retriever.main2(data_path,db_path,data_path+"/models/feverous_cell_extractor")
    f=open(data_path+"/fake_dev.jsonl","w")
    f.write(gen_fake_dev(data_path))
    f.close()
    baseline.retriever.predict_cells_from_table.main2(data_path+"/fake_dev.jsonl", db_path,data_path+"/models/feverous_cell_extractor")
    f_fd=open(data_path+"/fake_dev.cells.jsonl",'r')
    f_d=open(data_path+"/dev.jsonl",'r')

    d_obj=[]

    for line in f_d:
        d_obj+=[json.loads(line)]


    fd_obj=[]

    for line in f_fd:
        fd_obj+=[json.loads(line)]

    f_fd.close()
    f_d.close()
    res=''
    totalpres=0
    totalrecall=0
    nb=0
    for i in range(len(d_obj)):
        try:
            if(d_obj[i]['claim']!=fd_obj[i]['claim']):
                print('Skip: Different claims at index '+str(i))
            else:
                x=dict()
                x['claim']=d_obj[i]['claim']
                #print(x['claim'])
                gt_evidence=list(itertools.chain(*[x['content'] for x in d_obj[i]['evidence']]))
                pred_evidence=fd_obj[i]['predicted_evidence']
                pred_evidence=list(set(pred_evidence))




                common_evidence=[t for t in gt_evidence if t in pred_evidence]

                common_evidence=list(set(common_evidence))
                #print('common_evidence len:'+str(len(common_evidence)))
                onlygt_evidences=[t for t in gt_evidence if t not in pred_evidence]

                onlygt_evidences=list(set(onlygt_evidences))
                #print('onlygt_evidences len:'+str(len(onlygt_evidences)))
                onlypred_evidences=[t for t in pred_evidence if t not in gt_evidence]
                #print('onlypred_evidences len:'+str(len(onlypred_evidences)))
                onlypred_evidences=list(set(onlypred_evidences))


                common_evidence_v=[get_cell_value(t,db) for t in common_evidence]
                onlygt_evidences_v=[get_cell_value(t,db) for t in onlygt_evidences]
                onlypred_evidences_v=[get_cell_value(t,db) for t in onlypred_evidences]

                precision=len(common_evidence)/len(pred_evidence) if len(pred_evidence)>0 else 0
                recall=len(common_evidence)/len(gt_evidence) if len(gt_evidence)>0 else 1

                x['common_evidence']=common_evidence
                x['onlygt_evidences']=onlygt_evidences
                x['onlypred_evidences']=onlypred_evidences

                x['common_evidence_v']=common_evidence_v
                x['onlygt_evidences_v']=onlygt_evidences_v
                x['onlypred_evidences_v']=onlypred_evidences_v

                x['precision']=precision
                x['recall']=recall

                totalpres+=precision
                totalrecall+=recall
                nb+=1

                res+=json.dumps(x)+"\n"

        except Exception as e:
            print('Skip: Error on '+str(i))
            print('d:'+str(d_obj[i]))
            print('fd:'+str(fd_obj[i]))
            print(e)
            print(traceback.print_exc())

    #res+="\n\n\nprecision: " + str(totalpres/nb)+", recall: "+str(totalrecall/nb)

    f_comp=open(data_path+"/"+exp_name+'_retriever.jsonl','w')
    f_comp.write(res)
    f_comp.close()
    f_scores=open(data_path+"/"+exp_name+'_retriever_scores.json','w')
    dict_sc=dict()
    dict_sc['precision']=totalpres/nb
    dict_sc['recall']=totalrecall/nb
    f_scores.write(json.dumps(dict_sc))
    f_scores.close()
    shutil.rmtree(data_path+'/models')
    print("Retriever : Results written to " +data_path+"/"+exp_name+'.jsonl')


    
    
    
    
    

def exec_cell_retriever_alt_devdb(db_path, data_path, exp_name,db_path_test, dev_train, dev_test):
    db=FeverousDB(db_path)
    db_test=FeverousDB(db_path_test)
    f_d_t=open(data_path+"/dev.jsonl","w")
    f_d_s=open(data_path+"/"+dev_train,"r")
    for line in f_d_s:
        f_d_t.write(line)
    f_d_s.close()
    f_d_t.close()
    baseline.retriever.train_cell_evidence_retriever.main2(data_path,db_path,data_path+"/models/feverous_cell_extractor")
    
    f_d_t=open(data_path+"/dev.jsonl","w")
    f_d_s=open(data_path+"/"+dev_test,"r")
    for line in f_d_s:
        f_d_t.write(line)
    f_d_s.close()
    f_d_t.close()
    
    
    f=open(data_path+"/fake_dev.jsonl","w")
    f.write(gen_fake_dev(data_path))
    f.close()
    baseline.retriever.predict_cells_from_table.main2(data_path+"/fake_dev.jsonl", db_path_test,data_path+"/models/feverous_cell_extractor")
    f_fd=open(data_path+"/fake_dev.cells.jsonl",'r')
    f_d=open(data_path+"/dev.jsonl",'r')

    d_obj=[]

    for line in f_d:
        d_obj+=[json.loads(line)]


    fd_obj=[]

    for line in f_fd:
        fd_obj+=[json.loads(line)]

    f_fd.close()
    f_d.close()
    res=''
    totalpres=0
    totalrecall=0
    nb=0
    for i in range(len(d_obj)):
        try:
            if(d_obj[i]['claim']!=fd_obj[i]['claim']):
                print('Skip: Different claims at index '+str(i))
            else:
                x=dict()
                x['claim']=d_obj[i]['claim']
                #print(x['claim'])
                gt_evidence=list(itertools.chain(*[x['content'] for x in d_obj[i]['evidence']]))
                pred_evidence=fd_obj[i]['predicted_evidence']
                pred_evidence=list(set(pred_evidence))




                common_evidence=[t for t in gt_evidence if t in pred_evidence]

                common_evidence=list(set(common_evidence))
                #print('common_evidence len:'+str(len(common_evidence)))
                onlygt_evidences=[t for t in gt_evidence if t not in pred_evidence]

                onlygt_evidences=list(set(onlygt_evidences))
                #print('onlygt_evidences len:'+str(len(onlygt_evidences)))
                onlypred_evidences=[t for t in pred_evidence if t not in gt_evidence]
                #print('onlypred_evidences len:'+str(len(onlypred_evidences)))
                onlypred_evidences=list(set(onlypred_evidences))


                common_evidence_v=[get_cell_value(t,db_test) for t in common_evidence]
                onlygt_evidences_v=[get_cell_value(t,db_test) for t in onlygt_evidences]
                onlypred_evidences_v=[get_cell_value(t,db_test) for t in onlypred_evidences]

                precision=len(common_evidence)/len(pred_evidence) if len(pred_evidence)>0 else 0
                recall=len(common_evidence)/len(gt_evidence) if len(gt_evidence)>0 else 1

                x['common_evidence']=common_evidence
                x['onlygt_evidences']=onlygt_evidences
                x['onlypred_evidences']=onlypred_evidences

                x['common_evidence_v']=common_evidence_v
                x['onlygt_evidences_v']=onlygt_evidences_v
                x['onlypred_evidences_v']=onlypred_evidences_v

                x['precision']=precision
                x['recall']=recall

                totalpres+=precision
                totalrecall+=recall
                nb+=1

                res+=json.dumps(x)+"\n"

        except Exception as e:
            print('Skip: Error on '+str(i))
            print('d:'+str(d_obj[i]))
            print('fd:'+str(fd_obj[i]))
            print(e)
            print(traceback.print_exc())

    #res+="\n\n\nprecision: " + str(totalpres/nb)+", recall: "+str(totalrecall/nb)

    f_comp=open(data_path+"/"+exp_name+'_retriever.jsonl','w')
    f_comp.write(res)
    f_comp.close()
    f_scores=open(data_path+"/"+exp_name+'_retriever_scores.json','w')
    dict_sc=dict()
    dict_sc['precision']=totalpres/nb
    dict_sc['recall']=totalrecall/nb
    f_scores.write(json.dumps(dict_sc))
    f_scores.close()
    #shutil.rmtree(data_path+'/models')
    if  os.path.exists(data_path+'/models/feverous_cell_extractor/pytorch_model.bin'):
        os.remove(data_path+'/models/feverous_cell_extractor/pytorch_model.bin')
    if  os.path.exists(data_path+'/models/feverous_cell_extractor/config.json'):
        os.remove(data_path+'/models/feverous_cell_extractor/config.json')
    print("Retriever : Results written to " +data_path+"/"+exp_name+'.jsonl')


    
    
    
    

def exec_cell_retriever_alt_devdb_quick(db_path, data_path, exp_name,db_path_test, dev_train, dev_test,useSavedModel):
    db=FeverousDB(db_path)
    db_test=FeverousDB(db_path_test)
    f_d_t=open(data_path+"/dev.jsonl","w")
    f_d_s=open(data_path+"/"+dev_train,"r")
    for line in f_d_s:
        f_d_t.write(line)
    f_d_s.close()
    f_d_t.close()
    if not useSavedModel:
        if  os.path.exists(data_path+'/models/feverous_cell_extractor/pytorch_model.bin'):
            os.remove(data_path+'/models/feverous_cell_extractor/pytorch_model.bin')
        if  os.path.exists(data_path+'/models/feverous_cell_extractor/config.json'):
            os.remove(data_path+'/models/feverous_cell_extractor/config.json')
        baseline.retriever.train_cell_evidence_retriever.main2(data_path,db_path,data_path+"/models/feverous_cell_extractor")
    
    f_d_t=open(data_path+"/dev.jsonl","w")
    f_d_s=open(data_path+"/"+dev_test,"r")
    for line in f_d_s:
        f_d_t.write(line)
    f_d_s.close()
    f_d_t.close()
    
    
    f=open(data_path+"/fake_dev.jsonl","w")
    f.write(gen_fake_dev(data_path))
    f.close()
    baseline.retriever.predict_cells_from_table.main2(data_path+"/fake_dev.jsonl", db_path_test,data_path+"/models/feverous_cell_extractor")
    f_fd=open(data_path+"/fake_dev.cells.jsonl",'r')
    f_d=open(data_path+"/dev.jsonl",'r')

    d_obj=[]

    for line in f_d:
        d_obj+=[json.loads(line)]


    fd_obj=[]

    for line in f_fd:
        fd_obj+=[json.loads(line)]

    f_fd.close()
    f_d.close()
    res=''
    totalpres=0
    totalrecall=0
    nb=0
    for i in range(len(d_obj)):
        try:
            if(d_obj[i]['claim']!=fd_obj[i]['claim']):
                print('Skip: Different claims at index '+str(i))
            else:
                x=dict()
                x['claim']=d_obj[i]['claim']
                #print(x['claim'])
                gt_evidence=list(itertools.chain(*[x['content'] for x in d_obj[i]['evidence']]))
                pred_evidence=fd_obj[i]['predicted_evidence']
                pred_evidence=list(set(pred_evidence))




                common_evidence=[t for t in gt_evidence if t in pred_evidence]

                common_evidence=list(set(common_evidence))
                #print('common_evidence len:'+str(len(common_evidence)))
                onlygt_evidences=[t for t in gt_evidence if t not in pred_evidence]

                onlygt_evidences=list(set(onlygt_evidences))
                #print('onlygt_evidences len:'+str(len(onlygt_evidences)))
                onlypred_evidences=[t for t in pred_evidence if t not in gt_evidence]
                #print('onlypred_evidences len:'+str(len(onlypred_evidences)))
                onlypred_evidences=list(set(onlypred_evidences))


                common_evidence_v=[get_cell_value(t,db_test) for t in common_evidence]
                onlygt_evidences_v=[get_cell_value(t,db_test) for t in onlygt_evidences]
                onlypred_evidences_v=[get_cell_value(t,db_test) for t in onlypred_evidences]

                precision=len(common_evidence)/len(pred_evidence) if len(pred_evidence)>0 else 0
                recall=len(common_evidence)/len(gt_evidence) if len(gt_evidence)>0 else 1

                x['common_evidence']=common_evidence
                x['onlygt_evidences']=onlygt_evidences
                x['onlypred_evidences']=onlypred_evidences

                x['common_evidence_v']=common_evidence_v
                x['onlygt_evidences_v']=onlygt_evidences_v
                x['onlypred_evidences_v']=onlypred_evidences_v

                x['precision']=precision
                x['recall']=recall

                totalpres+=precision
                totalrecall+=recall
                nb+=1

                res+=json.dumps(x)+"\n"

        except Exception as e:
            print('Skip: Error on '+str(i))
            print('d:'+str(d_obj[i]))
            print('fd:'+str(fd_obj[i]))
            print(e)
            print(traceback.print_exc())

    #res+="\n\n\nprecision: " + str(totalpres/nb)+", recall: "+str(totalrecall/nb)

    f_comp=open(data_path+"/"+exp_name+'_retriever.jsonl','w')
    f_comp.write(res)
    f_comp.close()
    f_scores=open(data_path+"/"+exp_name+'_retriever_scores.json','w')
    dict_sc=dict()
    dict_sc['precision']=totalpres/nb
    dict_sc['recall']=totalrecall/nb
    f_scores.write(json.dumps(dict_sc))
    f_scores.close()
    #shutil.rmtree(data_path+'/models')
    
    print("Retriever : Results written to " +data_path+"/"+exp_name+'.jsonl')



