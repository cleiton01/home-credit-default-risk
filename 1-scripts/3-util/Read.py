import pandas as pd

class Read(object):

    read_data = {}
    """
        file = full path and file name that will be read.
    """
    def read_csv(file, sep=',', delimiter):

        if read_data.get(file) == None:
            data = pd.read_csv(filepath_or_buffer=file,sep=sep,delimiter=delimiter)
            read_data[file]=data
        else:
            data = read_data.get(file)
        
        return data
    
    def teste(self):
        
        
        for file in 'C:/`:
            self.exec_file(file)
            
    def exec_file(self, file):
        tmp_file = pd.read_csv(file)
        
        tmp_file['COD_CLIENTE'] = 0
        
        
    def check_valid_email(self,email):
        invalid_email = ['teste', 'tst', 'telefonica', 'vivo']
        valid_email = True 
        
        tmp = email.split('@')[1]
        email_to_validit = tmp.split('.')[0]
        
        if email_to_validit in invalid_email:
            valid_email = False
        return valid_email
    
    def get_cod_cliente(self, cnpj)
        cnpj_txt = cnpj.zfill(14)
        
        cod_cli = '40'+cnpj_txt[0:8]
        return cod_cli
    
    def check_cel_vivo(self, celular_number)
        
        con = Connection()
        
        sttm = con.get_oracle_coonection('REPLICA_BDA')
        
        cursor = sttm.cursor()
        
        cursor.execute('SELECT ACM.MSISDN, ACM.ACCOUNT, ACM.PROFILE, ACM.STATE, ACM.PARTITION_KEY FROM ACC_CLIENT_ACCOUNT CA JOIN ACC_CLIENT_MSISDN ACM WHERE ACM.MSISDN = {}'.format(celular_number))
        
        
        
        
        
        
        
        
    
    
    
    
    
