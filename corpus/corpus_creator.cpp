#include <bits/stdc++.h>

using namespace std;

#define MOD 1000000007
typedef long long ll;

int main()
{
    ll cnt = 0;

	vector<pair<ll, ll>> ind;

    string s;
    for(ll i = 0 ; i < 493593 ; i++)
    {
		cin >> s;
		string str;
		for(int j = 0 ; j < s.length() ; j++)
		{
	    	if(s.at(j) == ',' and j != 0)
				break;

	    	str += s.at(j);
		}

		if(str == ".")
			ind.push_back({0, i});
    }

	for(ll i = 1 ; i < ind.size() ; i++)
	{
		ind[i].first = ind[i - 1].second + 1;
	}

	for(ll i = 0 ; i < ind.size() ; i++)
	{
		cout << ind[i].first << "," << ind[i].second << endl;
	}

    return 0;
}

