import os
import re
import json
import json_repair






def get_files_in_path(directory):
    """
    递归获取指定路径下所有文件的绝对路径。
    """
    files_list = []
    for entry in os.scandir(directory):
        if entry.is_file():
            files_list.append(entry.path)
        elif entry.is_dir():
            files_list.extend(get_files_in_path(entry.path))
    return files_list


def load_json_file(input_file):
    if os.path.isfile(input_file):
        try:
            d_list = json.load(open(input_file, 'r', encoding='utf-8'))
        except:
            d_list =[json_repair.loads(line.strip()) for line in open(input_file, 'r', encoding='utf-8').readlines()]
    else:
        d_list_all=[]
        files=get_files_in_path(input_file)
        for file in files:
            try:
                d_list = json.load(open(file, 'r', encoding='utf-8'))
            except:
                d_list =[json_repair.loads(line.strip()) for line in open(file, 'r', encoding='utf-8').readlines()]
            d_list_all.extend(d_list)
        d_list=d_list_all
    return d_list


def write_json_file(output_file,d_output,is_line=True):
    if is_line:
        fw=open(output_file,'w',encoding='utf-8')
        for ii,d in enumerate(d_output):
            fw.write(json.dumps(d, ensure_ascii=False) + '\n')
        fw.close()
    else:
        json.dump(d_output,open(output_file,'w',encoding='utf-8'),ensure_ascii=False,indent=4)


def deep_sort(lst):
    """递归排序列表的所有层级"""
    # 先对当前列表排序
    lst.sort()
    # 然后检查列表中的每个元素，如果是列表就递归调用deep_sort
    for item in lst:
        if isinstance(item, list):
            deep_sort(item)
    return lst

    
def filter_json_text(text):
    text=text.replace('```','json_split_tag').replace('`',' ').replace('json_split_tag','```')
    pattern=r"```(json|Json|JSON|)\s+([^`]*)\s+```"
    match_obj=re.findall(pattern,text,re.DOTALL)
    match_list=[b for a,b in match_obj]
    if len(match_list)>0:
        return match_list[-1]
    return text


def simple_valid_func(predict_result):
    if predict_result == '-1':
        return False, 'predict_result==-1'

    predict_result = filter_json_text(predict_result)

    try:
        pred_d = json.loads(predict_result)
        # pred_d = json_repair.loads(predict_result)  
        return True, pred_d
    except:
        return False, 'error json predict_result'


def get_value_by_path(data, path):
    import re

    id=0
    while '[:]' in path:
        path=path.replace('[:]','[enum_all%d]'%id,1)
        id+=1
    # print(path)
    # 正则表达式，用于匹配键和索引
    pattern = r'\[([^\[\]]+)\]'
    # 将路径中的键和索引分解出来
    path=path.replace('"', "'")
    keys = re.findall(pattern, path)
    # 将所有的键转换为实际的整数索引或字符串键
    # keys = [(int(key) if key.isdigit() else key.strip("'")) for key in keys]

    # 逐步深入到数据结构中
    current = data
    for key in keys:
        if isinstance(current, (dict, list)):
            if isinstance(key, str) and 'enum_all' in key:
                return [get_value_by_path(data, path.replace('[%s]'%key,'[%d]'%ii)) for ii,ele in enumerate(current)]
            elif isinstance(key, str) and  '++' in key:
                ret_vals=[]
                for ele in key.split('++'):
                    val=get_value_by_path(data, path.replace("[%s]"%key,"[%s]"%ele))
                    if isinstance(val, list):
                        ret_vals.append(str(sorted(val)))
                    else:
                        ret_vals.append(str(val))
                return '++'.join(ret_vals)
            else:
                current = current[int(key)] if key.isdigit() else current[key.strip("'")]
        else:
            print('data:',[data])
            print('key:',key,keys)
            raise KeyError("路径中的键/索引无法在给定数据中找到")
    return current


def vote_result_from_file(input_file,output_file,group_id,predict_key,valid_func=simple_valid_func,count_by_path=None,threshold=0.5,step=0):
    d_list=load_json_file(input_file)
    d_list_new=[]
    if step!=0:
        for d in d_list:
            for pred in json.loads(d['pred_list']):
                d_new={**{k:v for k,v in d.items() if k!='pred_list'},**{predict_key:json.dumps(pred)}}
                d_list_new.append(d_new)
        d_list=d_list_new

    query2preds={}
    query2info={}
    invalid_num=0
    for ii,d in enumerate(d_list):
        query=d[group_id]
   
        predict_result=d[predict_key]
        is_valid,pred_d_new=valid_func(predict_result)
        if is_valid:
            query2info[query]=d
            if query not in query2preds:
                query2preds[query]=[]
            query2preds[query].append(pred_d_new)
        else:
            invalid_num+=1
            # if invalid_num<=30:
            #     print(['-----invalid predict_result',ii,d[group_id][:10],d['qid'],pred_d_new,json.dumps(predict_result,ensure_ascii=False)])
            
    print('total record num:%d',len(d_list))
    print('invalid predict_result num:%d',invalid_num)
    print('group_id num:%d',len(query2preds))

    d_output=[]
    for query,preds in query2preds.items():
        count_str2pred_list={}
        for pred in preds:
            # try:
            if count_by_path is None:
                count_str=count_str=json.dumps(pred, ensure_ascii=False)
            else:
                count_str=get_value_by_path(pred, path=count_by_path)
                count_str=deep_sort(count_str)
                count_str=json.dumps(count_str, ensure_ascii=False)
            if count_str not in count_str2pred_list:
                count_str2pred_list[count_str]=[]
            count_str2pred_list[count_str].append(pred)
            # except:
            #     pass
        
        count_str2cnt={count_str:len(pred_list) for count_str,pred_list in count_str2pred_list.items()}
        count_str_cnt=sorted([[count_str,cnt] for count_str,cnt in count_str2cnt.items()],key=lambda x:x[1],reverse=True)
        if len(count_str_cnt)==0:
            continue
        count_str_key=count_str_cnt[0][0]
        pred_list = json.dumps(count_str2pred_list[count_str_key],ensure_ascii=False)
        result=json.dumps(count_str2pred_list[count_str_key][0],ensure_ascii=False)
        pred_num=sum([tt[1] for tt in count_str_cnt]+[0])
        top_num=count_str2cnt[count_str_key]
        pct=top_num*1./pred_num if pred_num>0 else 0
        d_output.append({**query2info[query],**{'pred_num':pred_num,'top_num':top_num,'count_str_key':count_str_key,'pred_list':pred_list,predict_key:result,'pct':pct}})
    d_output=sorted(d_output,key=lambda x:x['pct'],reverse=True)
    print('after vote data num:%d',len(d_output))
    d_output=[d for d in d_output if d['pred_num']>=3 and d['pct']>=threshold]
    print('after filter threshold data num:%d',len(d_output))
    write_json_file(output_file,d_output,is_line=False)


def recursive_vote_result_from_file(input_file,output_file,group_id,predict_key,valid_func,step_count_by_path_threshold):
    for ii,p_d in enumerate(step_count_by_path_threshold):
        print('**************************step:',ii)
        count_by_path=p_d['count_by_path']
        threshold=p_d['threshold']
        input_file=input_file if ii==0 else output_file
        vote_result_from_file(input_file,output_file,group_id,predict_key,valid_func,count_by_path,threshold,ii)

##评测
def func_eval_f1(mark_pred_list):
    n_tp=0.
    n_t=0.
    n_p=0.
    for mark_list,pred_list in mark_pred_list:
        mark_list = [tt.lower() for tt in mark_list]
        pred_list = [tt.lower() for tt in pred_list]

        pred_list_new=[]
        for pred in pred_list:
            if pred not in pred_list_new:
                pred_list_new.append(pred)
        pred_list=pred_list_new
        if len(pred_list) == 0:
            pred_list = ['无答案']
        if len(mark_list) == 0:
            mark_list = ['无答案']

        pred_list_set=set(pred_list)
        mark_list_set=set(mark_list)

        n_t+=len(mark_list_set)
        n_p+=len(pred_list_set)
        n_tp+=len(pred_list_set&mark_list_set)
        # if pred_list_set!=mark_list_set:
        #     print(pred_list_set,mark_list_set,len(pred_list_set),len(mark_list_set),len(pred_list_set&mark_list_set))
    recall=n_tp/n_t if n_t>0 else 0.
    precison=n_tp/n_p if n_p>0 else 0.
    f1=2./(1./(n_tp/n_t)+1./(n_tp/n_p)) if n_tp>0 else 0.
    eval_ret={'recall':recall,'precison':precison,'f1_score':f1}
    # eval_ret='n_tp:%.1f n_t:%.1f n_p:%.1f recall:%.4f precison:%.4f f1:%.4f'%(n_tp,n_t,n_p,n_tp/n_t,n_tp/n_p,2./(1./(n_tp/n_t)+1./(n_tp/n_p)))
    return eval_ret

#pass评测 
def evaluation_with_path(file_path,json_path='',pred_key='output',mark_key='predict_result'):
    mark_pred_list=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if pred_key not in data:
                    data[pred_key]=data["choices"][0]["message"]["content"][0]["text"]
                # print(data[pred_key])
                output = json.loads(data[pred_key])
                if mark_key not in data:
                    data[mark_key]=data["choices"][0]["message"]["content"][0]["text"]
                predict_result = json.loads(data[mark_key])
                # print([output])
                v1 = get_value_by_path(output,json_path)
                # v1=[str(sorted(tttt)) if isinstance(tttt, list) else tttt for tttt in v1]       
                v2 = get_value_by_path(predict_result, json_path)
                # v2=[str(sorted(tttt)) if isinstance(tttt, list) else tttt for tttt in v2]       
                mark_pred_list.append([[v1],[v2]])
                # print([v1,v2])
            except:
                print("\n\n",json.dumps(data,ensure_ascii=False))

    eval_ret=func_eval_f1(mark_pred_list)
    return eval_ret


#pass评测 
def add_eval_result(file_path,json_path='',pred_key='output',mark_key='predict_result'):
    out=[]
    out_error=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if pred_key not in data:
                    data[pred_key]=data["choices"][0]["message"]["content"][0]["text"]
                # print(data[pred_key])
                output = json.loads(data[pred_key])
                if mark_key not in data:
                    data[mark_key]=data["choices"][0]["message"]["content"][0]["text"]
                predict_result = json.loads(data[mark_key])
                # print([output])
                v1 = get_value_by_path(output,json_path)
                v1=[str(sorted(tttt)) if isinstance(tttt, list) else tttt for tttt in v1]
                v2 = get_value_by_path(predict_result, json_path)
                v2=[str(sorted(tttt)) if isinstance(tttt, list) else tttt for tttt in v2]       
                data['eval_result']={"result":"True" if v1==v2 else "False"}
                out.append(data)
            except:
                out_error.append(data)
                pass
    with open(file_path,'w',encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o,ensure_ascii=False)+'\n') 
    print('++++++++error num:',len(out_error))
    with open(file_path+'.error.jsonl','w',encoding='utf-8') as f:
        for o in out_error:
            f.write(json.dumps(o,ensure_ascii=False)+'\n') 

def evaluation(input_path,**kwargs):
    # json_path='JSON["事件类型列表"][:]'
    json_path='JSON["事件类型"]'
    add_eval_result(input_path,json_path=json_path,pred_key='output',mark_key='predict_result')
    ret=evaluation_with_path(input_path,json_path=json_path,pred_key='output',mark_key='predict_result')
    return ret



if __name__=='__main__':
    pass