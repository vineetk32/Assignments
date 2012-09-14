1. Upload HelloWorld.c to your AFS through SFTP to remote-linux.eos.ncsu.edu.

(You can use WinSCP : http://winscp.net/eng/download.php to upload the source code, login with your unity account)

2. use SSH client (such as putty) to connect to

remote-linux.eos.ncsu.edu

and login with your unity account

(putty: http://www.chiark.greenend.org.uk/~sgtatham/putty/)

3. type: "add gcc346"

4. type: "g++ HelloWorld.c"

5. type: "./a.out"

6. the screen should print "Hello world!"

7. then try compiling basic search tree file