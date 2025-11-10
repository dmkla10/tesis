import pickle

sujetos = {1:{"edad":16, "comuna":"la pintana"},    # Grupo 1
         2:{"edad":15, "comuna":"la pintana"},      # Grupo 1
         3:{"edad":18, "comuna":"la pintana"},      # Grupo 1
         4:{"edad":17, "comuna":"la pintana"},      # Grupo 1 / 2
         5:{"edad":25, "comuna":"la granja"},       # Grupo 2 / 3
         6:{"edad":30, "comuna":"la granja"},       # Grupo 3
         7:{"edad":37, "comuna":"la granja"},       # Grupo 3
         8:{"edad":23, "comuna":"la pintana"},       # Grupo 4/5
         9:{"edad":25, "comuna":"la pintana"},       # Grupo 4
         10:{"edad":20, "comuna":"la pintana"},       # Grupo 4/5
         }

delitos = {1:([1,2,3,4], {"tipo":"encerrona", "lugar":"la granja", "agresión":1}),
           2:([1,3,4], {"tipo":"encerrona", "lugar":"la florida", "agresión":0}),
           3:([1,2,3], {"tipo":"encerrona", "lugar":"la florida", "agresión":1}),
           4:([1,2,3,4], {"tipo":"encerrona", "lugar":"la granja", "agresión":1}),
           5:([1,2,4], {"tipo":"encerrona", "lugar":"la granja", "agresión":1}),
           6:([1,2,3,4], {"tipo":"encerrona", "lugar":"la granja", "agresión":1}),
           7:([1,3,4], {"tipo":"encerrona", "lugar":"la florida", "agresión":0}),
           8:([1,2,3], {"tipo":"encerrona", "lugar":"la florida", "agresión":1}),
           9:([1,2,3,4], {"tipo":"encerrona", "lugar":"la granja", "agresión":1}),
           10:([1,2,4], {"tipo":"encerrona", "lugar":"la granja", "agresión":1}),
           11:([3,5], {"tipo":"asalto", "lugar":"la granja", "agresión":0}),
           12:([3,5], {"tipo":"asalto", "lugar":"la granja", "agresión":1}),
           13:([3,5], {"tipo":"asalto", "lugar":"la granja", "agresión":0}),
           14:([3,5], {"tipo":"asalto", "lugar":"la granja", "agresión":1}),
           15:([3], {"tipo":"asalto", "lugar":"la granja", "agresión":1}),
           16:([5,6,7], {"tipo":"encerrona", "lugar":"macul", "agresión":0}),
           17:([5,6,7], {"tipo":"encerrona", "lugar":"la florida", "agresión":0},),
           18:([5,6,7], {"tipo":"encerrona", "lugar":"macul", "agresión":0},),
           19:([5,6,7], {"tipo":"encerrona", "lugar":"macul", "agresión":0},),
           20:([8,9], {"tipo":"asalto", "lugar":"la granja", "agresión":0},),
           21:([8,9], {"tipo":"asalto", "lugar":"la granja", "agresión":1},),
           22:([8,9], {"tipo":"asalto", "lugar":"la granja", "agresión":0},),
           23:([8,10], {"tipo":"asalto", "lugar":"la granja", "agresión":1},),
           24:([8,10], {"tipo":"asalto", "lugar":"la granja", "agresión":1},),
           25:([1,3,10], {"tipo":"encerrona", "lugar":"la granja", "agresión":1}),
           26:([1,3,10], {"tipo":"encerrona", "lugar":"la pintana", "agresión":1}),
           27:([1,3,10], {"tipo":"encerrona", "lugar":"la pintana", "agresión":1}),
           }

# Saving the objects:
with open('data/simulated/data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([sujetos, delitos], f)