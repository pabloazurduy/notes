## Databases 
There are two types of databases: 
    1. relational: SQL, allow joins  
    2. non-relational (4 sub types)
        1. key-value stores 
        2. document stores
        3. column stores 
        4. graph stores

we use non-relational databases when one of these 4 conditions are met:
1. we need very low latency 
1. the data is non structured or there's no relational data 
1. you need to store a massive amount of data **

Database Scaling 
1. we usually have a higher ratio of reads vs writes. hence we usually have a master server and slaves, each slave just read and copy from the master, write/update operations are performed in the master node, reads are performed in the slave nodes. that way we can scalate horizaontally. 

## Scaling 
1. vertical scaling -> bigger machine (more CPU, RAM, bandwidth)
1. horizontal scaling -> more machines 

## Stateless and Stateful architectures 
In a stateless architecture the session data is not stored in the web-server but in a shared persistent evirorment. that means that if there's lose of one or many web servers the new requests from the client are semmesless redirected to a new server, given that the session data was not stored on those servers but in a shared space. 

In a Stateful arquitecture the session data is stored in the web server. hence, the LB does route all the request from a user directly to the same server, this mode is more vulnerable, given that if you lose a server the session data will be lost 

