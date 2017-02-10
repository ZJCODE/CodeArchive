sort_userhistorytable=userhistorytable.sort_values(['user_id','time'])
sort_userhistorytable.index = range(len(sort_userhistorytable))

ids=[sort_userhistorytable.user_id[0]]
times=[sort_userhistorytable.time[0]]
items=[[sort_userhistorytable.item_id[0]]]
for id_,item,time_ in sort_userhistorytable.ix[1:,:].values:
    if id_ == ids[-1]:
        if time_ == times[-1]:
            items[-1].append(item)
        else:
            ids.append(id_)
            times.append(time_)
            items.append([item])
    else:
        ids.append(id_)
        times.append(time_)
        items.append([item])

        
Drop = [len(a) for a in items]        
        
Users_Buy = pd.DataFrame({'ids':ids,'times':times,'items':items,'Drop':Drop})
    #print id_,time,item
Users_Buy_used = Users_Buy[Users_Buy.Drop >1][['ids','times','items']]