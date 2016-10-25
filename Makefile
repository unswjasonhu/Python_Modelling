test:
	nose2 -v
coverage:
	nose2 --with-coverage  --coverage-report html -vv
mysql:
	mysql -hdb -upollution -ppollution pollution_monitoring < /code/src/sql_statements.sql

