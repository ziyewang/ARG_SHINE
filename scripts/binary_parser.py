#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2017/6/28
@author yrh

"""

import os
import sys
import json
import configparser
from collections import defaultdict

#from pfp.utils import get_pid_sp

__all__ = ['get_binary_feature']

def get_pid_sp(file):
    pid_sp = {}
    with open(file) as fp:
        for line in fp:
            pid, sp = line.split()[:2]
            pid_sp[pid] = sp
    return pid_sp

def create_domain_table(table_file, interpro_out_files, pid_sp=None, sp=None):
    domain_set = set()
    for file in interpro_out_files:
        with open(file) as fp:
            data = json.load(fp)
            """
            New version of output JSON files by InterProScan
            """
            if isinstance(data, dict) and 'results' in data:
                data = data['results']
            for seq in data:
                flag, pid_list = False, []
                try:
                    for cr in seq['crossReferences']:
                        pid_list.append(cr['identifier'])
                except KeyError:
                    for xref in seq['xref']:
                        pid_list.append(xref['id'])
                for pid in pid_list:
                    if pid_sp is None or sp is None or (pid in pid_sp and pid_sp[pid] in sp):
                        flag = True
                if flag:
                    for domain in seq['matches']:
                        domain_set.add(domain['signature']['accession'])
        print(file)
    with open(table_file, 'w') as fp:
        for domain in domain_set:
            print(domain, file=fp)


def get_domain_table(table_file):
    table = {}
    with open(table_file) as fp:
        for domain_id, line in enumerate(fp):
            table[line.strip()] = domain_id
    return table


def convert(feature_file, interpro_out_file, table, pid_sp=None, sp=None):
    with open(interpro_out_file) as fp:
        data = json.load(fp)
        """
        New version of output JSON files by InterProScan
        """
        if isinstance(data, dict) and 'results' in data:
            data = data['results']
    pid_set = set()
    try:
        with open(feature_file) as fp:
            for line in fp:
                pid_set.add(line.split()[0])
    except FileNotFoundError:
        pass
    with open(feature_file, 'a') as fp:
        for seq in data:
            domain_id_set = set()
            for domain in seq['matches']:
                accession = domain['signature']['accession']
                if accession in table:
                    domain_id_set.add(table[accession])
            features = [str(idx) + ':1' for idx in domain_id_set]
            try:
                for cr in seq['crossReferences']:
                    cr_id = cr['identifier']
                    if ((cr_id not in pid_set) and
                            (pid_sp is None or sp is None or (cr_id in pid_sp and pid_sp[cr_id] in sp))):
                        print(cr_id, *features, file=fp)
                        pid_set.add(cr_id)
            except KeyError:
                for xref in seq['xref']:
                    pid = xref['id']
                    if ((pid not in pid_set) and
                            (pid_sp is None or sp is None or (pid in pid_sp and pid_sp[pid] in sp))):
                        print(pid, *features, file=fp)
                        pid_set.add(pid)


def get_binary_feature(feature_file):
    f = defaultdict(list)
    with open(feature_file) as fp:
        for line in fp:
            pid, *f_ = line.split()
            if f_:
                f[pid] += [int(x.split(':')[0]) + 1 for x in f_]
    return f


def main(argv):
    conf = configparser.ConfigParser()
    print(conf.read(argv))
    if conf.has_option('interpro', 'species'):
        sp = set(conf.get('interpro', 'species').split())
        pid_sp = get_pid_sp(conf.get('interpro', 'protein_species'))
    else:
        sp = pid_sp = None
    if not os.path.exists(conf.get('interpro', 'table')) or conf.getboolean('create', 'create', fallback=False):
        create_domain_table(conf.get('interpro', 'table'),
                            conf.get('create', 'interpro_output').split(),
                            pid_sp, sp)
    if conf.getboolean('convert', 'convert', fallback=True):
        feature_file, table = conf.get('interpro', 'feature'), get_domain_table(conf.get('interpro', 'table'))
        for interpro_out_file in conf.get('convert', 'interpro_output').split():
            convert(feature_file, interpro_out_file, table, pid_sp, sp)
            print(interpro_out_file)


if __name__ == '__main__':
    main(sys.argv[1:])
