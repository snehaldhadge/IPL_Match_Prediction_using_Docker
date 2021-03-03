USE IPL;

DROP TABLE IF EXISTS game;

CREATE TABLE game as
SELECT id,(CASE WHEN m.city = t1.Home_Ground or m.city = t1.Second_Ground then team1
				WHEN m.city = t2.Home_Ground or m.city = t2.Second_Ground then team2
				ELSE team1 END) as Home_team,
	(CASE WHEN m.city = t1.Home_Ground or m.city = t1.Second_Ground then team2
				WHEN m.city = t2.Home_Ground or m.city = t2.Second_Ground then team1
				ELSE team2 END) as Away_team
FROM matches m
JOIN teams t1 ON
t1.Team_name = m.team1
JOIN teams t2 ON
t2.Team_name = m.team2
order by `date`;



DROP TABLE IF EXISTS game_stats;

CREATE TABLE game_stats as
SELECT g.*,season,city,`date`,
(CASE WHEN toss_winner = Home_Team THEN 1 ELSE 0 END) as Home_Toss_Won,
(CASE WHEN toss_winner = Home_Team AND toss_decision ='bat' THEN 1 ELSE 0 END) as Home_Toss_Bat,
(CASE WHEN winner = Home_Team  THEN 1 ELSE 0 END) as Home_Team_Win,
(CASE WHEN toss_winner = Away_Team THEN 1 ELSE 0 END) as Away_Toss_Won,
(CASE WHEN toss_winner = Away_Team AND toss_decision ='bat' THEN 1 ELSE 0 END) as Away_Toss_Bat,
(CASE WHEN winner = Away_Team  THEN 1 ELSE 0 END) as Away_Team_Win
FROM matches m
JOIN game g on g.id = m.id
order by `date`;




# Getting match level statistics

DROP TABLE IF EXISTS Team_Match_Stats;

CREATE TABLE  Team_Match_Stats as
SELECT d.match_id,inning,d.batting_team,d.bowling_team,sum(total_runs) as Total_Runs,COUNT(ball) as Total_Balls,
	   max(`over`) as Overs, sum(wide_runs) as Total_wide,sum(bye_runs) as Bye_Runs,
	   sum(legbye_runs) as Leg_Bye,sum(noball_runs) as Noball_runs,
	   sum(penalty_runs) as penalty_runs,sum(extra_runs) as Extra_runs,Number_Wickets,Number_4s,Number_6s,Dot_Balls,
	   (sum(total_runs)- sum(extra_runs))/balls_faced as Batting_StrikeRate,
	   (sum(total_runs)- sum(extra_runs))/Number_Wickets as BattingAverage,
	   count(ball)/Number_Wickets as Bowling_SR,
	   runs_conceded/Number_Wickets as Bowling_Avg,
	   sum(total_runs)/max(`over`) as Bowling_Econ
FROM deliveries d
LEFT JOIN (SELECT match_id,batting_team,COUNT(player_dismissed) as Number_Wickets FROM deliveries d2 where player_dismissed != ''
GROUP BY match_id,batting_team)
d2 ON d.match_id = d2.match_id and d2.batting_team = d.batting_team
LEFT  JOIN (SELECT match_id,batting_team,count(1) as Number_4s  FROM deliveries  where  batsman_runs = 4
GROUP BY match_id,batting_team)
d4 ON d.match_id = d4.match_id and d4.batting_team = d.batting_team
LEFT  JOIN (SELECT match_id,batting_team,count(1) as Number_6s  FROM deliveries where  batsman_runs = 6
GROUP BY match_id,batting_team)
d3 ON d.match_id = d3.match_id and d3.batting_team = d.batting_team
LEFT  JOIN (SELECT match_id,batting_team,count(1) as Dot_Balls  FROM deliveries where  batsman_runs=0 and extra_runs=0
GROUP BY match_id,batting_team)
d5 ON d.match_id = d5.match_id and d5.batting_team = d.batting_team
LEFT  JOIN (SELECT match_id,batting_team,count(ball) as balls_faced FROM deliveries where  wide_runs =0
GROUP BY match_id,batting_team)
d6 ON d.match_id = d6.match_id and d6.batting_team = d.batting_team
LEFT  JOIN (SELECT match_id,batting_team,sum(total_runs) as runs_conceded FROM deliveries where  bye_runs =0
and legbye_runs =0 and penalty_runs =0
GROUP BY match_id,batting_team)
d7 ON d.match_id = d7.match_id and d7.batting_team = d.batting_team
GROUP BY match_id,batting_team
ORDER by match_id;



drop table if exists Team_Match_Stats_1;

CREATE TABLE Team_Match_Stats_1
select t.*,
(BattingAverage * Batting_StrikeRate) as Batting_Index,
(Bowling_Avg * Bowling_SR) as Bowling_Index
from Team_Match_Stats t
group by match_id,inning ;


# Batsman scoring 100 and 50

DROP TABLE  IF EXISTS batsman_stats;

CREATE table batsman_stats as
SELECT match_id,inning,batting_team,bowling_team ,batsman,
	   sum(batsman_runs) as total_runs,count(ball) as total_balls
FROM deliveries d2
WHERE wide_runs=0
GROUP BY
batsman,match_id,inning
ORDER BY match_id,inning,batsman;




# get Number of players who scored 100 and 50 from each match
# Centuries

# Create table Century Fifty Stats


DROP TABLE IF EXISTS Cent_FIF_stats;



CREATE TABLE Cent_FIF_stats as
select t.*,IFNULL(Century ,0) as Century,IFNULL(Fifty ,0) as Fifty
FROM Team_Match_Stats_1 t LEFT JOIN
(SELECT match_id,batting_team,Count(1) as Century
from batsman_stats
where total_runs >= 100
group by match_id,batting_team) b
ON b.match_id = t.match_id and b.batting_team = t.batting_team
 LEFT JOIN
(SELECT match_id,batting_team,Count(1) as Fifty
from batsman_stats
where total_runs >= 50 and total_runs<100
group by match_id,batting_team) f
ON f.match_id = t.match_id and f.batting_team = t.batting_team
ORDER BY t.match_id,t.batting_team;





DROP TABLE IF EXISTS Each_Match_Stats;

CREATE TABLE Each_Match_Stats as
SELECT c.*,m.`Date` FROM Cent_FIF_stats c
JOIN matches m
ON c.match_id = m.id;



# Create Rolling Batting Stats for each team
DROP TABLE IF EXISTS Rolling_Batting_Averages_Team;



CREATE TABLE Rolling_Batting_Averages_Team
SELECT match_id,`Date`,batting_team as Team,
AVG(Batting_Index) OVER (PARTITION BY batting_team ORDER BY `Date` ASC,batting_team  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Batting_Index,
AVG(Bowling_Econ) OVER (PARTITION BY  batting_team  ORDER BY `Date` ASC,batting_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Run_Rate,
AVG(Number_4s) OVER (PARTITION BY batting_team ORDER BY `Date` ASC,batting_team  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Number_4s,
AVG(Number_6s) OVER (PARTITION BY batting_team ORDER BY `Date` ASC,batting_team  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Number_6s,
COUNT(match_id) OVER (PARTITION BY batting_team ORDER BY `Date` ASC,batting_team  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Matches_Played,
AVG(Century) OVER (PARTITION BY batting_team ORDER BY `Date` ASC,batting_team  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Century,
AVG(Fifty) OVER (PARTITION BY batting_team ORDER BY `Date` ASC,batting_team  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Fifty
FROM Each_Match_Stats f
GROUP BY match_id, batting_team
ORDER BY `Date`, match_id,batting_team;





# Rolling Team Bowling Average
DROP TABLE IF EXISTS Rolling_Bowling_team_Averages;



CREATE TABLE Rolling_Bowling_team_Averages
SELECT match_id,`Date`,bowling_team as Team,
AVG(Bowling_Econ) OVER (PARTITION BY bowling_team ORDER BY `Date` ASC,bowling_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Bowling_Econ,
AVG(Bowling_Index) OVER (PARTITION BY bowling_team ORDER BY `Date` ASC,bowling_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Bowling_Index,
AVG(Extra_runs) OVER (PARTITION BY bowling_team ORDER BY `Date` ASC,bowling_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as AVG_Extras,
AVG(overs) OVER (PARTITION BY bowling_team ORDER BY `Date` ASC,bowling_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Avg_Overs,
AVG(Total_Balls) OVER (PARTITION BY bowling_team ORDER BY `Date` ASC,bowling_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Avg_balls
FROM Each_Match_Stats
GROUP BY match_id, bowling_team
ORDER BY `Date` ASC,match_id, bowling_team;

# Rolling Team1 Toss and Bat decision

DROP TABLE IF EXISTS Rolling_game_stats;

CREATE TABLE Rolling_game_stats
SELECT id,season,`date`,city,Home_team,Away_team, Home_team_Win,
AVG(Home_Toss_Won) OVER (PARTITION BY Home_team ORDER BY `date` ASC,Home_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Home_Avg_Toss_Won,
AVG(Home_Toss_Bat) OVER (PARTITION BY Home_team ORDER BY `date` ASC,Home_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Home_Avg_Bat_Won,
AVG(Home_team_Win) OVER (PARTITION BY Home_team ORDER BY `date` ASC,Home_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Home_Avg_Wins,
AVG(Away_Toss_Won) OVER (PARTITION BY Away_team ORDER BY `date` ASC,Away_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Away_Avg_Toss_Won,
AVG(Away_Toss_Bat) OVER (PARTITION BY Away_team ORDER BY `date` ASC,Away_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Away_Avg_Bat_Won,
AVG(Away_team_Win) OVER (PARTITION BY Away_team ORDER BY `date` ASC,Away_team ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) as Away_Avg_Wins
FROM game_stats
ORDER by `date` ;




# Club all stats for Home Team
DROP TABLE IF EXISTS Home_total_stats;


CREATE TABLE Home_total_stats as
select a.*, Batting_Index as Home_BI,
	   Run_rate as Home_RR,
       Number_4s as Home_4s, Number_6s as Home_6s, Matches_Played as Home_matchplayed,
       Century as Home_Centuries, Fifty as Home_Fifties, Bowling_Index as Home_BoI,
       Bowling_Econ as Home_BoEcon,
       AVG_Extras as Home_Extras
FROM Rolling_game_stats a
JOIN Rolling_Batting_Averages_Team b
ON a.id = b.match_id and a.Home_team = b.Team
JOIN Rolling_Bowling_team_Averages c
ON a.id = c.match_id and a.Home_team = c.Team
ORDER by `date`,a.id;



# Add Away Team stats

DROP TABLE IF EXISTS Final_Match_Stats;



CREATE TABLE Final_Match_Stats as
select a.*, Batting_Index as  Away_BI,
	   Run_rate as Away_RR,
       Number_4s as Away_4s, Number_6s as Away_6s, Matches_Played as Away_matchplayed,
       Century as Away_Centuries, Fifty as Away_Fifties, Bowling_Index as Away_BoI,
       Bowling_Econ as Away_BoEcon,
       AVG_Extras as Away_Extras
FROM Home_total_stats a
JOIN Rolling_Batting_Averages_Team b
ON a.id = b.match_id and a.Away_team = b.Team
JOIN Rolling_Bowling_team_Averages c
ON a.id = c.match_id and a.Away_team = c.Team
ORDER by `date`,a.id;




update Final_Match_Stats
   SET Home_team=(SELECT team_id FROM teams t WHERE Home_team=t.Team_name ),
   Away_team=(SELECT team_id FROM teams t WHERE Away_team=t.Team_name );



DROP TABLE IF EXISTS Final_diff_stats;

CREATE TABLE Final_diff_stats as
SELECT id,season,Home_team,Away_team,Home_team_Win,
(Home_BI - Away_BI) as Batting_Index,
(Home_RR - Away_RR) as Net_Run_Rate,
(Home_4s - Away_4s) as Fours,
(Home_6s - Away_6s) as Six,
(Home_matchplayed - Away_matchplayed) as matchplayed,
(Home_Centuries - Away_Centuries) as Centuries,
(Home_Fifties - Away_Fifties) as Fifties,
(Home_BoI- Away_BoI) as Bowling_Index,
(Home_Extras - Away_Extras) as Extras,
(Home_BoEcon - Away_BoEcon) as Bo_Econ,
(Home_Avg_Toss_Won - Away_Avg_Toss_Won) as Avg_Toss_Won,
(Home_AVG_Bat_Won - Away_AVG_Bat_Won) as AVG_Bat_Won,
(Home_AVG_Wins - Away_AVG_Wins) as AVG_Wins
FROM Final_Match_Stats
ORDER BY `date` asc;


DROP TABLE IF EXISTS Final_ratio_stats;

CREATE TABLE Final_ratio_stats as
SELECT id,season,Home_team,Away_team,Home_team_Win,
(Home_BI /NULLIF(Away_BI,0)) as Batting_Index,
(Home_4s /NULLIF(Away_4s,0)) as Fours,
(Home_6s /NULLIF(Away_6s,0)) as Six,
(Home_matchplayed /NULLIF(Away_matchplayed,0)) as matchplayed,
(Home_Centuries /NULLIF(Away_Centuries,0)) as Centuries,
(Home_Fifties/NULLIF(Away_Fifties,0)) as Fifties,
(Home_BoI/NULLIF(Away_BoI,0)) as Bowling_Index,
(Home_Extras /NULLIF(Away_Extras,0)) as Extras,
(Home_Avg_Toss_Won /NULLIF(Away_Avg_Toss_Won,0)) as Avg_Toss_Won,
(Home_AVG_Bat_Won /NULLIF(Away_AVG_Bat_Won,0)) as AVG_Bat_Won,
(Home_AVG_Wins/NULLIF(Away_AVG_Wins,0)) as AVG_Wins
FROM Final_Match_Stats
ORDER BY `date` asc;

