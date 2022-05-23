-- Query 1 & 2: Find the “hot” securities. (SQL)

-- Aims: Find securities with high turnover & high range (We name them “hot securities.”)

-- QUERY 1:  find the Sector which has the biggest turnover over the period. 
SELECT sum(a.close * a.volume) as turnover, b.GICSSector
	FROM equity_prices as a left join equity_static as b
		on a.symbol_id = b.symbol	
	GROUP BY b.GICSSector
	ORDER BY turnover DESC
	;
-- According to the result, we know Information Technology has the biggest turnover. 


-- QUERY 2: find the "hot" security.
CREATE TABLE hot_security as 
SELECT 
	avg(close), max(close) - min(close) as range, sum(a.close * a.volume) as turnover, b.security
	FROM equity_prices as a left join equity_static as b
		on a.symbol_id = b.symbol
	WHERE b.GICSSector like "I% T%" -- we can also ues WHERE b.GICSSector in ("Information Technology")
	GROUP BY b.security
	HAVING  (range > avg(close) * 1) and turnover > avg(a.close * a.volume) 
	ORDER BY turnover DESC, RANGE DESC
	;
-- According to the result, we get 17 hot securities, which are worthy of further investigation. 
	