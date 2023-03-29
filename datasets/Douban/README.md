We collected users' ratings in three domains(i.e., movie, book and music) from Douban(www.douban.com), which is a popular review website in China.
The statistics of Douban datasets are summarized as follows:

| Dataset     | #user  | #item   | #event     |
|-------------|--------|---------|------------|
| DoubanMovie | 94,890 | 81,906  | 11,742,260 |
| DoubanMusic | 39,742 | 164,223 | 1,792,501  |
| DoubanBook  | 46,548 | 212,995 | 1,908,081  |

Besides rating data, we also crawled the social connections between users.

|           | #node   | #edge     |
|-----------|---------|-----------|
| SocialNet | 695,800 | 1,758,302 |

*Note*: In three datasets, we replace a blank rating with -1. A blank rating indicates that a user has an interaction wth the item but doesn't rate it.

This new dataset can support various kinds of research on recommender systems, such as ***social recommendation***, ***dynamic recommendation*** and ***multi-domain recommendation***. 

If you use these datasets in your research, please cite our paper:
```
@inproceedings{song2019session,
  title={Session-Based Social Recommendation via Dynamic Graph Attention Networks},
  author={Song, Weiping and Xiao, Zhiping and Wang, Yifan and Charlin, Laurent and Zhang, Ming and Tang, Jian},
  booktitle={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  pages={555--563},
  year={2019},
  organization={ACM}
}
```
