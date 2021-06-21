
class DBMSUtils(object):
    available_dbms = ['sqlserver', 'mysql']

    @staticmethod
    def get_dbms_from_str_connection(str_connection: str):
        assert isinstance(str_connection, str), "Wrong data type for param 'str_connection'."

        dbms = None
        if 'mysql' in str_connection:
            dbms = 'mysql'
        elif 'SQL Server' in str_connection:
            dbms = 'sqlserver'

        return dbms

    @staticmethod
    def check_dbms(name: str):
        assert isinstance(name, str), "Wrong data type for param 'name'."
        assert name in DBMSUtils.available_dbms, f"No dbms {name} found. Use one of {DBMSUtils.available_dbms}."

        return name

    @staticmethod
    def get_delimited_col(dbms: str, col: str):
        dbms = DBMSUtils.check_dbms(dbms)
        assert isinstance(col, str), "Wrong data type for param 'col'."

        if dbms == 'mysql':
            return f'`{col}`'

        if dbms == 'sqlserver':
            return f'[{col}]'

    @staticmethod
    def create_index(dbms: str, index_name: str, target_table: str, target_col: str):
        dbms = DBMSUtils.check_dbms(dbms)
        assert isinstance(index_name, str), "Wrong data type for param 'index_name'."
        assert isinstance(target_table, str), "Wrong data type for param 'target_table'."
        assert isinstance(target_col, str), "Wrong data type for param 'target_col'."

        if dbms == 'mysql':
            return f"\nCREATE INDEX {index_name} on {target_table}({target_col});\n"

        if dbms == 'sqlserver':
            return f"\nCREATE INDEX {index_name} on {target_table}({target_col});\n"    # FIXME: to check

    @staticmethod
    def get_auto_increment_col(dbms: str):
        dbms = DBMSUtils.check_dbms(dbms)

        if dbms == 'mysql':
            return f"ROW_NUMBER() OVER(ORDER BY (SELECT NULL))"

        if dbms == 'sqlserver':
            return f"ROW_NUMBER() OVER(ORDER BY (SELECT NULL))"

    @staticmethod
    def get_expression_limits(dbms: str):
        dbms = DBMSUtils.check_dbms(dbms)

        if dbms == 'mysql':
            return 5000              # FIXME: to implement

        if dbms == 'sqlserver':
            return 2965
