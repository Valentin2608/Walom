# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:17:56 2020

@author: u
"""

import numpy as np
import matplotlib.pyplot as plt 
import random
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy.matlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

Nb_test=5000
Nb_predi=200
nb_parametres=6
essai=1
y_cotefaible=np.ones(shape=(Nb_test,))
y_train=np.ones(shape=(Nb_test,))
x_train=np.ones(shape=(Nb_test,nb_parametres))
y_test=np.ones(shape=(Nb_predi,))
x_test=np.ones(shape=(Nb_predi,nb_parametres))
My_predi=np.ones(shape=(Nb_predi,))
bankroll=np.zeros(shape=(Nb_test+1,))
match=[]
SurfaceJ1=[]
SurfaceJ2=[]
RankJ1=[]
RankJ2=[]
PointJ1=[]
PointJ2=[]
listeNomJoueurGagnant=[]
listeNomJoueurPerdant=[]
HomeJ1=[]
HomeJ2=[]
NomJ1=[]
NomJ2=[]
cotej1=[]
cotej2=[]
tab_num_joueur_gagnant=[]
winrate=0
tab80=np.zeros(shape=(Nb_predi,))
compteur=0
bankroll_t=[]
winrate_t=[]
for m in range (0,essai):
    compteur=0
    winrate=0
    with open('tennisbetVF22.csv', newline='') as csv_file:
           data = csv.reader(csv_file, delimiter=';', quotechar='|')
           #on met dans la liste toutes les valeurs de clotures existantes sur le fichier
           for colonne in data:
               NomJ1.append(colonne[9])
               NomJ2.append(colonne[10])
               SurfaceJ1.append(colonne[11])
               SurfaceJ2.append(colonne[12])
               RankJ1.append(colonne[17])
               RankJ2.append(colonne[18])
               PointJ1.append(colonne[19])
               PointJ2.append(colonne[20])
               HomeJ1.append(colonne[13])
               HomeJ2.append(colonne[14])
               cotej1.append(colonne[40])
               cotej2.append(colonne[41])
               
    with open('2018.csv', newline='') as csv_file:
           data = csv.reader(csv_file, delimiter=';', quotechar='|')
          
        
           #on met dans la liste toutes les valeurs de clotures existantes sur le fichier
           for colonne in data:
               NomJ1.append(colonne[9])
               NomJ2.append(colonne[10])
               SurfaceJ1.append(colonne[11])
               SurfaceJ2.append(colonne[12])
               RankJ1.append(colonne[15])
               RankJ2.append(colonne[16])
               PointJ1.append(colonne[17])
               PointJ2.append(colonne[18])
               HomeJ1.append(colonne[13])
               HomeJ2.append(colonne[14])
               cotej1.append(colonne[42])
               cotej2.append(colonne[43])
          
    del NomJ1 [2595]   
    del NomJ2 [2595]  
    del SurfaceJ1 [2595]   
    del SurfaceJ2 [2595]  
    del RankJ1 [2595]  
    del RankJ2 [2595]  
    del PointJ1 [2595]  
    del PointJ2 [2595]  
    del HomeJ1 [2595]  
    del HomeJ2 [2595]  
    del cotej1 [2595]  
    del cotej2 [2595]  
    
    
    for i in range(1,Nb_test+Nb_predi+1):
        val=int(PointJ1[i])
        val2=int(PointJ2[i])
        
        
        if val<=560:
            RankJ1[i]=1
        elif  val<=920:
            RankJ1[i]=2
        elif  val<=1460:
            RankJ1[i]=3
        elif  val <=2000:
            RankJ1[i]=4
        elif  val >2000:
            RankJ1[i]=5
            
            
        if val2<=560:
            RankJ2[i]=1
        elif val2<=920:
            RankJ2[i]=2
        elif val2<=1460:
            RankJ2[i]=3
        elif val2<=2000:
            RankJ2[i]=4
        elif val2>2000:
            RankJ2[i]=5
          
            #on remplit la matrice x_train avec les valeurs souhaitée
    for i in range(0,Nb_test)   : 
              
               k=i+1
               num_joueur_gagnant= random.randint(1, 2)
               tab_num_joueur_gagnant.append(num_joueur_gagnant)
               if num_joueur_gagnant==1:
                
                 
                   x_train[i,0]= int(SurfaceJ1[k])
                   x_train[i,1]=int(SurfaceJ2[k])
                   x_train[i,2]=int(PointJ1[k])
                   x_train[i,3]=int(PointJ2[k])
                   x_train[i,4]=int(HomeJ1[k])
                   x_train[i,5]=int(HomeJ2[k])
                   """
                   x_train[i,6]=int(RankJ1[k])
                   x_train[i,7]=int(RankJ2[k])
                   """
                   """
                   x_train[i,6]=float(cotej1[k].replace(",","."))
                   x_train[i,7]=float(cotej2[k].replace(",","."))
                   """
                  # x_train[i,5]=float(cotej2[k].replace(",","."))
                 
                   y_train[i,]=num_joueur_gagnant
                  
               elif num_joueur_gagnant==2:
                   x_train[i,1]= int(SurfaceJ1[k])
                   x_train[i,0]=int(SurfaceJ2[k])
                   x_train[i,3]=int(PointJ1[k])
                   x_train[i,2]=int(PointJ2[k])
                   x_train[i,5]=int(HomeJ1[k])
                   x_train[i,4]=int(HomeJ2[k])
                   """
                   x_train[i,7]=int(RankJ1[k])
                   x_train[i,6]=int(RankJ2[k])
                   """
                   """
                   x_train[i,7]=float(cotej1[k].replace(",","."))
                   x_train[i,6]=float(cotej2[k].replace(",","."))
                   """
                   y_train[i,]=num_joueur_gagnant
                   
                   
    
    for i in range(0,Nb_predi):
               k=i+Nb_test+1
               num_joueur_gagnant= random.randint(1, 2)
               tab_num_joueur_gagnant.append(num_joueur_gagnant)
               if num_joueur_gagnant==1:
                   x_test[i,0]= int(SurfaceJ1[k])
                   x_test[i,1]=int(SurfaceJ2[k])
                   x_test[i,2]=int(PointJ1[k])
                   x_test[i,3]=int(PointJ2[k])
                   x_test[i,4]=int(HomeJ1[k])
                   x_test[i,5]=int(HomeJ2[k])
                   """
                   x_test[i,6]=int(RankJ1[k])
                   x_test[i,7]=int(RankJ2[k])
                   """
                   """
                   x_test[i,6]=float(cotej1[k].replace(",","."))
                   x_test[i,7]=float(cotej2[k].replace(",","."))
                   """
                   y_test[i,]= num_joueur_gagnant
                   match.append([NomJ1[k],NomJ2[k],float(cotej1[k].replace(",",".")),float(cotej2[k].replace(",","."))])
                   
               elif num_joueur_gagnant==2:
                   x_test[i,1]= int(SurfaceJ1[k])
                   x_test[i,0]=int(SurfaceJ2[k])
                   x_test[i,3]=int(PointJ1[k])
                   x_test[i,2]=int(PointJ2[k])
                   x_test[i,5]=int(HomeJ1[k])
                   x_test[i,4]=int(HomeJ2[k])
                   """
                   x_test[i,7]=int(RankJ1[k])
                   x_test[i,6]=int(RankJ2[k])
                   """
                   """
                   x_test[i,7]=float(cotej1[k].replace(",","."))
                   x_test[i,6]=float(cotej2[k].replace(",","."))
                   """
                 
                   y_test[i,]=num_joueur_gagnant
                   match.append([NomJ2[k],NomJ1[k],float(cotej2[k].replace(",",".")),float(cotej1[k].replace(",","."))])
                  
                   
                   
    model= RandomForestClassifier(n_estimators=90)
    model.fit(x_train,y_train)  
    print(model.score(x_train, y_train))
    for i in range (0,Nb_predi):
        if int(match[i][2])<int(match[i][3]):
            My_predi[i]=1
        else:
            My_predi[i]=2
        
    y_predi2=model.predict(x_test)
    for i in range (0,Nb_predi):
       if float(cotej1[i+Nb_test+1].replace(",","."))<1.50 or float(cotej2[i+Nb_test+1].replace(",","."))<1.50:
                # if model.predict_proba(x_test[i,:].reshape(1,nb_parametres))[0][0]>=0.50 or  model.predict_proba(x_test[i,:].reshape(1,nb_parametres))[0][0]<=0.50  :
                    tab80[i]=My_predi[i]
                    
           
        
    
    for i in range(0,Nb_predi):
            if tab80[i]!=0:
                if tab80[i]==y_test[i,]:
                    winrate+=1
                    compteur+=1
                    print("Pari gagné : ",match[i][int(tab80[i])-1], "cote de : ",match[i][int(tab80[i])+1])
                    cote1=float(cotej1[i+Nb_test+1].replace(",","."))
                    
                    bankroll[compteur]=bankroll[compteur-1]+10*cote1-10
                
                else:
                    compteur+=1
                    print("Pari perdu : ",match[i][int(y_test[i,])-1], "cote de : ",match[i][int(y_test[i,])+1])
                    bankroll[compteur]=bankroll[compteur-1]-10
            else:
                pass
                print("Pari pas joue  : ",match[i][0],"contre",match[i][1]," cote de :",match[i][2],"contre ",match[i][3])
    del SurfaceJ1 [0:]   
    del SurfaceJ2 [0:]  
    del RankJ1 [0:]  
    del RankJ2 [0:]  
    del PointJ1 [0:]  
    del PointJ2 [0:]  
    del HomeJ1 [0:]  
    del HomeJ2 [0:]  
    del cotej1 [0:]  
    del cotej2 [0:]  
    bankroll_t.append(bankroll[compteur])
    winrate_t.append(winrate/compteur)
    print("winrate de :",winrate/compteur) 
    

print("bénéfice moyen final sur",essai, "essai: ",sum(bankroll_t)/essai)
print("Winrate moyen final sur",essai, "essai: ",sum(winrate_t)/essai)
plt.plot(bankroll_t[0:-1])
plt.title("Courbe des gains")
plt.xlabel("Nombre de paris")
plt.ylabel("Gain(en €)")
plt.show()
    






"""
def winner(model,surfacej1,surfacej2,rankj1,rankj2,agej1,agej2,homej1,homej2,pointsj1,pointsj2):
    x=np.array([surfacej1-surfacej2,abs(agej1-27),abs(agej2-27),rankj1-rankj2,pointsj1-pointsj2,homej1-homej2]).reshape(1,6)
    joueur1=0
    joueur2=0
   
    for i in range(1000):
        model.fit(x_train,y_train)
        print(model.score(x_train, y_train))
        joueur1=model.predict_proba(x)[0][0]+joueur1
        joueur2=model.predict_proba(x)[0][1]+joueur2
    
    print(joueur1)
    print(joueur2)
   



winner(model,1,0,13, 113, 32, 27, 0,1, 2713, 655)
 
 
"""  
    
    
    
    
    
    
    
    
    