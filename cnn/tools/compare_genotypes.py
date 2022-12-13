import json
import sys
import os

def same_genotype(geno, geno_list):
#  if geno in geno_list: return True
#  else: return False
  test_normal = geno['normal']
  test_reduce = geno['reduce']
  for d in geno_list:
    ref_normal = d['normal']
    ref_reduce = d['reduce']
    is_same = True
    for i in range(len(test_normal)//2):
      # test normal
      ref = ref_normal[2*i:2*i+2]
      test = test_normal[2*i:2*i+2]
      for tmp in test:
        if tmp not in ref: 
          is_same=False
          break
      if is_same==False: break
      # test reduce
      ref = ref_reduce[2*i:2*i+2]
      test = test_reduce[2*i:2*i+2]
      for tmp in test:
        if tmp not in ref: 
          is_same=False
          break
      if is_same==False: break
    if is_same: return True
  return False

def get_distinct_genotype(genotype_path, names=None):
  distinct_genotype = []
  distinct_names = []
  if names is None:
      names = range(len(os.listdir(genotype_path)))
  for name in names:
      genotype_file = os.path.join(genotype_path, '%s.txt'%name)
      tmp_dict = json.load(open(genotype_file,'r'))
      is_same = same_genotype(tmp_dict, distinct_genotype)
      if not is_same:
        distinct_genotype.append(tmp_dict)
        if name>10:
          distinct_names.append(name)
  return distinct_genotype, distinct_names


if __name__ == '__main__':
  base_dir = 'ckpt/search-EXP-20201210-142421-task0'
  genotype_path = os.path.join(base_dir, 'results_of_7q/genotype')
  names = None
#  names = [i for i in range(19)]
#  names.extend([i for i in range(29,50)])
  distinct_genotypes, distinct_names = get_distinct_genotype(genotype_path, names)
  print(len(distinct_genotypes))
  print(distinct_names)
#  for i in range(len(genotypes)):
#    print(names[i], genotypes[i])
