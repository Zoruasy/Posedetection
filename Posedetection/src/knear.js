export default class kNear {
    constructor(k, training = []) {
        this.k = k
        this.training = training;
        if (this.training.length === 0) {
            this.array_size = -1;
        } else {
            this.array_size = this.training[0].v.length;
        }
    }


    dist(v1, v2) {
        let sum = 0
        v1.forEach( (val, index) => {
            sum += Math.pow(val - v2[index], 2)
        })
        return Math.sqrt(sum)
    };

    updateMax(val, arr) {
        let max = 0
        for(let obj of arr) {
            max = Math.max(max, obj.d)
        }
        return max
    }

    mode(store) {
        let frequency = {}
        let max = 0
        let result
        for (let v in store) {
            frequency[store[v]] = (frequency[store[v]] || 0) + 1;
            if (frequency[store[v]] > max) {
                max = frequency[store[v]]
                result = store[v]
            }
        }
        return result
    }

    checkInput(v) {
        if (Array.isArray(v)) {

            if (v.length > 0) {

                if (typeof v[0] == 'number') {

                    if (this.array_size > -1) {

                        if (v.length == this.array_size) {

                            return true;
                        } else {
                            console.error(`Learn en classify verwachten een array met numbers van dezelfde lengte, je stuurt nu een array met lengte ${v.length}, terwijl je eerder lengte ${this.array_size} gebruikt hebt.`);
                        }
                    } else {

                        this.array_size = v.length;
                        return true;
                    }
                } else {
                    console.error(`Learn en classify verwachten een array met numbers, je stuurt nu array met ${typeof v[0]}.`);
                }
            } else {
                console.error("Learn en classify verwachten een array met numbers, je stuurt nu lege array.");
            }
        } else {
            console.error(`Learn en classify verwachten een array met numbers, je stuurt nu geen array, maar ${typeof v}.`);
        }


        return false
    }




    learn(vector, label) {
        this.checkInput(vector)
        let obj = { v: vector, lab: label }
        this.training.push(obj)
    }


    classify(v) {
        this.checkInput(v)
        let voteBloc = []
        let maxD = 0

        for(let obj of this.training) {
            let o = { d: this.dist(v, obj.v), vote: obj.lab }
            if (voteBloc.length < this.k) {
                voteBloc.push(o);
                maxD = this.updateMax(maxD, voteBloc)
            } else {
                if (o.d < maxD) {
                    let bool = true
                    let count = 0
                    while (bool) {
                        if (Number(voteBloc[count].d) === maxD) {
                            voteBloc.splice(count, 1, o)
                            maxD = this.updateMax(maxD, voteBloc)
                            bool = false
                        } else {
                            if (count < voteBloc.length - 1) {
                                count++
                            } else {
                                bool = false
                            }
                        }
                    }
                }
            }
        }
        let votes = []
        for(let el of voteBloc) {
            votes.push(el.vote)
        }
        return this.mode(votes)
    }
}