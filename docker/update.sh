#/bin/bash
# Change the filepath of openml.file
# from "https://www.openml.org/data/download/1666876/phpFsFYVN"
# to "http://minio:9000/datasets/0000/0001/phpFsFYVN"
mysql -hdatabase -uroot -pok -e 'UPDATE openml.file SET filepath = CONCAT("http://minio:9000/datasets/0000/", LPAD(id, 4, "0"), "/", SUBSTRING_INDEX(filepath, "/", -1)) WHERE extension="arff";'

# Update openml.expdb.dataset with the same url
mysql -hdatabase -uroot -pok -e 'UPDATE openml_expdb.dataset DS, openml.file FL SET DS.url = FL.filepath WHERE DS.did = FL.id;'





# Create the data_feature_description TABLE. TODO: can we make sure this table exists already?
mysql -hdatabase -uroot -pok -Dopenml_expdb -e 'CREATE TABLE IF NOT EXISTS `data_feature_description` (
  `did` int unsigned NOT NULL,
  `index` int unsigned NOT NULL,
  `uploader` mediumint unsigned NOT NULL,
  `date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `description_type` enum("plain", "ontology") NOT NULL,
  `value` varchar(256) NOT NULL,
  KEY `did` (`did`,`index`),
  CONSTRAINT `data_feature_description_ibfk_1` FOREIGN KEY (`did`, `index`) REFERENCES `data_feature` (`did`, `index`) ON DELETE CASCADE ON UPDATE CASCADE
)'

# SET dataset 1 to active (used in unittests java)
mysql -hdatabase -uroot -pok -Dopenml_expdb -e 'INSERT IGNORE INTO dataset_status VALUES (1, "active", "2024-01-01 00:00:00", 1)'
mysql -hdatabase -uroot -pok -Dopenml_expdb -e 'DELETE FROM dataset_status WHERE did = 2 AND status = "deactivated";'

# Temporary fix in case the database missed the kaggle table. The PHP Rest API expects the table to be there, while indexing.
mysql -hdatabase -uroot -pok -Dopenml_expdb -e 'CREATE TABLE IF NOT EXISTS `kaggle` (`dataset_id` int(11) DEFAULT NULL, `kaggle_link` varchar(500) DEFAULT NULL)'
