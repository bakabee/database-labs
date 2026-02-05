# Lab 1 Learning Notes
 
## Environment Setup
- Used native terminal (Mac/Linux)
- PostgreSQL 15 running smoothly
- DBeaver connected after fixing authentication issue

## Key Learnings
 
### Technical Skills 
1. **PostgreSQL Installation:** 
      Learned how to install PostgreSQL on Linux Mint and connect it via DBeaver
2. **SQL Basics:**
      Learned creating tables, inserting data, using SELECT , WHERE, GROUP BY, ORDER BY, and date functions.
3. **Git Workflow:**
      Learned git add to stage files, git commit to save changes locally, and git push to upload to GitHub

### Challenges Faced
1. **Challenge:** 
      DBeaver connection with FATAL: password authentication failed
- **Solution:**
Created or reset PostgreSQL user password and used correct credentials in DBeaver
- **Lesson:**
Always verify database users exist and passwords match

2. **Challenge:**
      pg_dump command run inside SQL terminal didnot schema file
- **Solution:**
Ran pg_dump in regular terminal instead of psql
- **Lesson:**
Distinguish between SQL commands and terminal commands

3. **Challenge:**
      GitHub authenication failed when pushing username/password
- **Solution:**
Generated SSH key and used SSH remote to push changes
- **Lesson:**
Modern Github requires tokens or SSH; username/password alone often fails

### AI Usage Reflection
- Used AI 10 times
- Most helpful: 7th interaction -leaning to fix PostgreSQL pager (END) issue and return to normal prompt
- Least helpful: 9th interaction(Github Push) -initial didn't solve authentication issue immediately
- Verification method: Tested commands in terminal/psql and confirmed results matched expected outputs

## Next STeps 
- Explore more SQL functions
- Practice git branching and merging workflows
- Learn about database indexes to optimize queries           
