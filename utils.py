import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import caffe
import lmdb
from caffe.io import datum_to_array, array_to_datum
from caffe.proto import caffe_pb2




def classes_hist(filename,columns,col,flag='05submask',classes_dist=False):
    """
    dataset classes histogram, uses irma code or 05 submask
    export into csv as train/test labels
    """

    if flag =='05submask':
        #working on 05 submask
        categories = pd.read_excel(filename,parse_cols=columns)
        categories.columns=['id','code']

    else:
        categories = pd.read_excel(filename,parse_cols=columns)
        categories.columns=['id','code']

    categories['code'].apply(str);
    image_id=categories['id']
    categories=categories['code'].str.split('-',expand=True)
    categories = categories[2]
    categories = categories.astype(str)
    categories = categories[categories != 'None']
    categories = categories.str[:1]
    #categories.value_counts().plot(kind="bar",color=col)
    categories_value_count = categories.value_counts()
    if classes_dist:
        print categories_value_count
        print 'sum %d'%categories.value_counts().sum()

    return pd.concat([image_id.astype(str),categories],axis=1)



def subclasses_hist(filename):
    """
    dataset subclasses histograms
    """
    categories = pd.read_excel(filename,parse_cols="a,b")
    categories.columns=['id ','code']
    categories['code'].apply(str);
    categories=categories['code'].str.split('-',expand=True)
    categories = categories[2]
    categories = categories.astype(str)
    categories = categories[categories != 'None']
    plt.suptitle('subcategories distribution')
    for i in range(2,10):
        subcategory=categories.copy()
        subcategory = subcategory[subcategory.str[0] ==str(i)]
        plt.subplot(8,2,i-1)
        subcategory.value_counts().plot(kind="bar",color='blue')
        subcategory.value_counts()
    plt.savefig('subcatogories_distribution.png')



def resize_images(img_dir,sorted_images_names):
    os.chdir(img_dir)
    for i in xrange(len(sorted_images_names)):
        img = cv2.imread(sorted_images_names[i],cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(64,64))
        cv2.imwrite(sorted_images_names[i],img)





def sort_names(a,size,flag ='with_extension'):
    """
    sort images names
    """
    c=[]
    for i  in xrange(size):
        b=a[i]
        x,y=b.split('.')
        a[i]=x

    a=np.asarray(a).astype(int)
    a=np.sort(a)

    for i  in xrange(size):
        b=str(a[i])
        b=b+'.png'
        c.append(b)
    if flag == 'with_extension':
        return c
    elif flag =='without_extension':
        return a




def compute_mean_image(img_dir,train_labels):
    """
    compute mean image
    """
    files_names=os.listdir(img_dir)
    files_sorted= sort_names(files_names,len(files_names),flag='without_extension')

    images_vectorized = np.zeros((1,64*64),dtype='float32')

    for i in range(len(files_names)):
            if i%1000 ==0:
                print 'in %d'%i
            if int(train_labels.loc[train_labels['id']==files_sorted[i]]['class'].values)==1 or  \
            int(train_labels.loc[train_labels['id']==files_sorted[i]]['class'].values)==6 or  \
            int(train_labels.loc[train_labels['id']==files_sorted[i]]['class'].values)==7 or  \
             int(train_labels.loc[train_labels['id']==files_sorted[i]]['class'].values)==8:
                continue
            else:
                image_name = img_dir+'/'+str(files_sorted[i])+'.png'
                temp_image = cv2.imread(image_name,cv2.IMREAD_GRAYSCALE)
                temp_image=temp_image.flatten()
                images_vectorized=np.row_stack((images_vectorized,temp_image))

    images_vectorized=images_vectorized[1:]
    mean_image = np.mean(images_vectorized,axis=0)/np.std(images_vectorized,axis=0)
    mean_image=np.reshape(mean_image,(64,64))
    #plt.imshow(mean_image,cmap='gray')
    return mean_image


def write_images_to_lmdb(files_sorted,random_index,labels):
    """
    write training and validation images and labels into lmdb format
    """

    print '---------- creating training db map ----------'

    old_labels = [2,3,4,5]
    db_name = 'train_data'
    train_map_size = 64*64*32*len(files)
    train_env = lmdb.Environment('/output/'+db_name, map_size=train_map_size)
    train_txn = train_env.begin(write=True,buffers=True)


    print '---------- initializing Writing ----------'
    for idx in range(10000):
        if int(labels.loc[labels['id']==files_sorted[random_index[idx]]]['class'].values)==1  or \
        int(labels.loc[labels['id']==files_sorted[random_index[idx]]]['class'].values)==6 or \
        int(labels.loc[labels['id']==files_sorted[random_index[idx]]]['class'].values)==7 or \
        int(labels.loc[labels['id']==files_sorted[random_index[idx]]]['class'].values)==8:
            continue
        else:
            image_name = str(files_sorted[random_index[idx]])+'.png'
            X = caffe.io.load_image(os.path.join(root, image_name), color=False)
            #X = cv2.resize(X,(64,64))
            X=caffe.io.resize_image(X,(64,64))
            X=np.divide(X-X.min(),X.max()-X.min()) #-----------------------
            #X=X-mean_image
            X = np.transpose(X,(2,0,1))
            y = labels['class'][random_index[idx]]

            if y in old_labels:
                #y1=y-2
                y-=2
            else:
                #y1=4
                y=4

            #print '%d  >  %d'%(y,y1)

            if files_sorted[random_index[idx]] == labels['id'][random_index[idx]]:
                if idx%1000 ==0:
                    print 'in %d'%idx
                    print X.min(),X.max()
                datum = array_to_datum(X,y)
                str_id = '{:08}'.format(idx)
                train_txn.put(str_id.encode('ascii'), datum.SerializeToString())

    train_txn.commit()
    train_env.close()
    print "---------- Writing training db done ----------"




def write_images_to_lmdb3(files_sorted,random_index,labels):
    print '---------- write training and validation images and labels into lmdb format ----------'

    print '---------- creating validation db map ----------'
    old_labels = [2,3,4,5]

    db_name = 'validation_data'
    validation_map_size = 64*64*32*len(files)
    validation_env = lmdb.Environment('/output/'+db_name, map_size=validation_map_size)
    validation_txn = validation_env.begin(write=True,buffers=True)

    print '---------- initializing Writing ----------'
    for idx in range(10000,11744):
        if int(labels.loc[labels['id']==files_sorted[random_index[idx]]]['class'].values)==1  or \
        int(labels.loc[labels['id']==files_sorted[random_index[idx]]]['class'].values)==6 or \
        int(labels.loc[labels['id']==files_sorted[random_index[idx]]]['class'].values)==7 or \
        int(labels.loc[labels['id']==files_sorted[random_index[idx]]]['class'].values)==8:
            continue
        else:
            image_name = str(files_sorted[random_index[idx]])+'.png'
            X = caffe.io.load_image(os.path.join(root, image_name), color=False)
            #X = cv2.resize(X,(64,64))
            X=caffe.io.resize_image(X,(64,64))
            X=np.divide(X-X.min(),X.max()-X.min()) #-----------------------
            #X=X-mean_image
            X = np.transpose(X,(2,0,1))
            y = labels['class'][random_index[idx]]

            if y in old_labels:
                #y1=y-2
                y-=2
            else:
                #y1=4
                y=4

            #print '%d  >  %d'%(y,y1)

            if files_sorted[random_index[idx]] == labels['id'][random_index[idx]]:
                if idx%1000 ==0:
                    print 'in %d'%idx
                    print X.min(),X.max()
                datum = array_to_datum(X,y)
                str_id = '{:08}'.format(idx)
                validation_txn.put(str_id.encode('ascii'), datum.SerializeToString())

    validation_txn.commit()
    validation_env.close()
    print "---------- Writing validation db done ----------"

def write_images_to_lmdb2(img_dir, db_name,labels):
    """
    write test images and labels into lmdb format
    """
    print '---------- write training images and labels into lmdb format ----------'

    for root, dirs, files in os.walk(img_dir, topdown = False):
        print '---------- searching for image directory ----------'
        if root != img_dir:
            continue
        print '---------- creating db map ----------'
        map_size = 64*64*32*len(files)
        env = lmdb.Environment(db_name, map_size=map_size)
        txn = env.begin(write=True,buffers=True)
        files.remove('.floydignore')
        files.remove('.floyddata')
        files.remove('.floydexpt')

        files_sorted=sort_names(files,len(files),flag='without_extension')

        print '---------- initializing Writing ----------'
        for idx in range(len(files_sorted)):
            if int(labels.loc[labels['id']==files_sorted[idx]]['class'].values)==1 or  \
            int(labels.loc[labels['id']==files_sorted[idx]]['class'].values)==6 or      \
            int(labels.loc[labels['id']==files_sorted[idx]]['class'].values)==7 or       \
            int(labels.loc[labels['id']==files_sorted[idx]]['class'].values)==8:
                continue
            else:
                image_name = str(files_sorted[idx])+'.png'
                X = caffe.io.load_image(os.path.join(root, image_name), color=False)
                #X = cv2.resize(X,(64,64))
                X=caffe.io.resize_image(X,(64,64))

                X = np.transpose(X,(2,0,1))
                y = int(labels.loc[labels['id']==files_sorted[idx]]['class'].values)

                if files_sorted[idx] == int(labels.loc[labels['id']==files_sorted[idx]]['id'].values):
                    if idx%1000 ==0:
                        print 'in %d'%idx
                    datum = array_to_datum(X,y)
                    str_id = '{:08}'.format(idx)
                    txn.put(str_id.encode('ascii'), datum.SerializeToString())
    txn.commit()
    env.close()
    print "---------- Writing training db done ----------"






def read_images_from_lmdb(db_name, visualize):
    """
    read images from lmbd dataset
    """
    env = lmdb.open(db_name)
    txn = env.begin()
    cursor = txn.cursor()
    X = []
    y = []
    idxs = []
    for idx, (key, value) in enumerate(cursor):
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        X.append(np.array(datum_to_array(datum)))
        y.append(datum.label)
        idxs.append(idx)
    if visualize:
        print "Visualizing a few images..."
        for i in range(10):
            img = X[i]
            plt.subplot(5,5,i+1)
            plt.imshow(img[0,:,:],cmap='gray')
            plt.title(y[i])
            plt.axis('off')
        plt.show()
    print " ".join(["Reading from", db_name, "done!"])





if __name__ == '__main__':
    #generate train lmbd from dataset
    #floyd run --data g4YAok4Ztjrc5SsLRUgyvT:label --data zwztervSStoJuVdEnHEjaa:training_data  --env caffe:py2 "python utils.py"
    img_dir='/training_data/'
    #img_dir = "/media/markeb/DC2CF7092CF6DE08/graduation_project/dataset2009/ImageCLEFmed2009_train.02/"
    for root, dirs, files in os.walk(img_dir, topdown = False):
        print '---------- searching for image directory ----------'
        if root != img_dir:
            continue
    #print files

    files.remove('.floydignore')
    files.remove('.floyddata')
    files.remove('.floydexpt')
    files_sorted=sort_names(files,len(files),flag="without_extension")
    random_index=np.arange(0,len(files_sorted))
    np.random.shuffle(random_index)

    train_labels=pd.read_csv('/label/train_labels.csv',delimiter=',')
    #train_labels=pd.read_csv('/media/markeb/DC2CF7092CF6DE08/graduation_project/dataset2009/train_labels.csv',delimiter=',')
    train_labels=train_labels[['id','class']]
    print '---------- training labels acquired ----------'

    write_images_to_lmdb(files_sorted,random_index,train_labels)
    write_images_to_lmdb3(files_sorted,random_index,train_labels)

    experiment = "dataset size : 11744\nclasses : 2,3,4,5,9\nsize: 64x64\nnormalized to range [0,1]"
    np.savetxt('/output/description.txt',[experiment],fmt='%s')

    '''
    #generate lmdb tet from dataset
    img_dir = "/testing"
    db_name = "test_data"
    test_labels=pd.read_csv('/label/test_labels.csv',delimiter=',')
    test_labels=test_labels[['id','class']]
    print '---------- testing labels acquired ----------'
    write_images_to_lmdb2(img_dir,db_name,test_labels)
    experiment = "dataset size : full \nclasses : 2,3,4,5,9 \nsize: 64x64"
	np.savetxt('description.txt',[experiment],fmt='%s')
    '''







"""
example usage on pc
os.chdir('/media/markeb/DC2CF7092CF6DE08/graduation_project/dataset2009/')

img_dir = 'ImageCLEFmed2009_train.02'
images_names = os.listdir(img_dir)
sorted_images_names = sort_names(images_names,len(images_names))
resize_images(img_dir,sorted_images_names)


img_dir = "/media/markeb/DC2CF7092CF6DE08/graduation_project/dataset2009/ImageCLEFmed2009_train.02/"
train_labels=pd.read_csv('/media/markeb/DC2CF7092CF6DE08/graduation_project/dataset2009/train_labels.csv',delimiter=',')
train_labels=train_labels[['id','class']]
mean_image = compute_mean_image(img_dir,train_labels)
plt.imshow(mean_image,cmap='gray')

img_dir = "/media/markeb/DC2CF7092CF6DE08/graduation_project/dataset2009/ImageCLEFmed2009_train.02/"
db_name = "training_data"
train_labels=pd.read_csv('/media/markeb/DC2CF7092CF6DE08/graduation_project/dataset2009/train_labels.csv',delimiter=',')
train_labels=train_labels[['id','class']]
write_images_to_lmdb(img_dir,db_name,train_labels,mean_image)



img_dir = "/media/markeb/DC2CF7092CF6DE08/graduation_project/dataset2009/ImageCLEFmed2009_test.0/"
db_name = "test_data3"
test_labels=pd.read_csv('/media/markeb/DC2CF7092CF6DE08/graduation_project/dataset2009/test_labels.csv',delimiter=',')
test_labels=test_labels[['id','class']]
write_images_to_lmdb2(img_dir,db_name,test_labels,mean_image)


read_images_from_lmdb('/media/markeb/DC2CF7092CF6DE08/graduation_project/dataset2009/training_data/',True)
"""
