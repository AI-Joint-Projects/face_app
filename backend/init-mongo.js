// init-mongo.js
db = db.getSiblingDB('reaco_db');
db.createUser({
  user: 'user',
  pwd: 'password',
  roles: [{ role: 'readWrite', db: 'reaco_db' }]
});
