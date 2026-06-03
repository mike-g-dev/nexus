use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nexus_fix_codec::{
    FixDate, FixDecimal, FixMonthYear, FixTenor, FixTime, FixTimestamp, FixTzTime, FixTzTimestamp,
    encode_fix_char, encode_fix_int, encode_fix_seqnum, encode_fix_text, encode_fix_uint,
    parse_fix_char, parse_fix_int, parse_fix_multi_char, parse_fix_multi_string, parse_fix_seqnum,
    parse_fix_text, parse_fix_uint,
};

fn bench_fix_decimal(c: &mut Criterion) {
    let mut g = c.benchmark_group("FixDecimal::parse");

    g.bench_function("4_digit_price", |b| {
        b.iter(|| FixDecimal::parse(black_box(b"99.50")))
    });

    g.bench_function("8_digit_price", |b| {
        b.iter(|| FixDecimal::parse(black_box(b"50123.450")))
    });

    g.bench_function("12_digit_price", |b| {
        b.iter(|| FixDecimal::parse(black_box(b"50123.45000000")))
    });

    g.bench_function("16_digit_price", |b| {
        b.iter(|| FixDecimal::parse(black_box(b"1234567.890123456")))
    });

    g.bench_function("integer_only", |b| {
        b.iter(|| FixDecimal::parse(black_box(b"12345678")))
    });

    g.bench_function("negative", |b| {
        b.iter(|| FixDecimal::parse(black_box(b"-123.456")))
    });

    g.bench_function("sub_penny", |b| {
        b.iter(|| FixDecimal::parse(black_box(b"0.00000001")))
    });

    g.finish();
}

fn bench_fix_int(c: &mut Criterion) {
    let mut g = c.benchmark_group("parse_fix_int");

    g.bench_function("1_digit", |b| b.iter(|| parse_fix_int(black_box(b"7"))));

    g.bench_function("4_digit", |b| b.iter(|| parse_fix_int(black_box(b"1234"))));

    g.bench_function("8_digit", |b| {
        b.iter(|| parse_fix_int(black_box(b"12345678")))
    });

    g.bench_function("16_digit", |b| {
        b.iter(|| parse_fix_int(black_box(b"1234567890123456")))
    });

    g.bench_function("19_digit_max", |b| {
        b.iter(|| parse_fix_int(black_box(b"9223372036854775807")))
    });

    g.bench_function("negative_8", |b| {
        b.iter(|| parse_fix_int(black_box(b"-12345678")))
    });

    g.finish();
}

fn bench_fix_uint(c: &mut Criterion) {
    let mut g = c.benchmark_group("parse_fix_uint");

    g.bench_function("body_length", |b| {
        b.iter(|| parse_fix_uint(black_box(b"256")))
    });

    g.bench_function("num_in_group", |b| {
        b.iter(|| parse_fix_uint(black_box(b"12")))
    });

    g.finish();
}

fn bench_fix_seqnum(c: &mut Criterion) {
    let mut g = c.benchmark_group("parse_fix_seqnum");

    g.bench_function("small", |b| b.iter(|| parse_fix_seqnum(black_box(b"1000"))));

    g.bench_function("typical", |b| {
        b.iter(|| parse_fix_seqnum(black_box(b"1000000")))
    });

    g.bench_function("large", |b| {
        b.iter(|| parse_fix_seqnum(black_box(b"99999999999")))
    });

    g.finish();
}

fn bench_fix_timestamp(c: &mut Criterion) {
    let mut g = c.benchmark_group("FixTimestamp::parse");

    g.bench_function("no_frac", |b| {
        b.iter(|| FixTimestamp::parse(black_box(b"20260602-14:30:00")))
    });

    g.bench_function("millis", |b| {
        b.iter(|| FixTimestamp::parse(black_box(b"20260602-14:30:00.123")))
    });

    g.bench_function("micros", |b| {
        b.iter(|| FixTimestamp::parse(black_box(b"20260602-14:30:00.123456")))
    });

    g.bench_function("nanos", |b| {
        b.iter(|| FixTimestamp::parse(black_box(b"20260602-14:30:00.123456789")))
    });

    g.finish();
}

fn bench_fix_date(c: &mut Criterion) {
    c.bench_function("FixDate::parse", |b| {
        b.iter(|| FixDate::parse(black_box(b"20260602")))
    });
}

fn bench_fix_time(c: &mut Criterion) {
    let mut g = c.benchmark_group("FixTime::parse");

    g.bench_function("no_frac", |b| {
        b.iter(|| FixTime::parse(black_box(b"14:30:00")))
    });

    g.bench_function("micros", |b| {
        b.iter(|| FixTime::parse(black_box(b"14:30:00.123456")))
    });

    g.finish();
}

// -- Encode benchmarks --

fn bench_encode_decimal(c: &mut Criterion) {
    let mut g = c.benchmark_group("FixDecimal::encode");

    let dec_4 = FixDecimal::parse(b"99.50").unwrap();
    let dec_8 = FixDecimal::parse(b"50123.450").unwrap();
    let dec_16 = FixDecimal::parse(b"1234567.890123456").unwrap();
    let dec_int = FixDecimal::parse(b"12345678").unwrap();
    let dec_neg = FixDecimal::parse(b"-123.456").unwrap();

    g.bench_function("4_digit_price", |b| {
        b.iter(|| {
            let mut buf = [0u8; 22];
            black_box(dec_4).encode(black_box(&mut buf))
        })
    });

    g.bench_function("8_digit_price", |b| {
        b.iter(|| {
            let mut buf = [0u8; 22];
            black_box(dec_8).encode(black_box(&mut buf))
        })
    });

    g.bench_function("16_digit_price", |b| {
        b.iter(|| {
            let mut buf = [0u8; 22];
            black_box(dec_16).encode(black_box(&mut buf))
        })
    });

    g.bench_function("integer_only", |b| {
        b.iter(|| {
            let mut buf = [0u8; 22];
            black_box(dec_int).encode(black_box(&mut buf))
        })
    });

    g.bench_function("negative", |b| {
        b.iter(|| {
            let mut buf = [0u8; 22];
            black_box(dec_neg).encode(black_box(&mut buf))
        })
    });

    g.finish();
}

fn bench_encode_int(c: &mut Criterion) {
    let mut g = c.benchmark_group("encode_fix_int");

    g.bench_function("8_digit", |b| {
        b.iter(|| {
            let mut buf = [0u8; 20];
            encode_fix_int(black_box(12_345_678), black_box(&mut buf))
        })
    });

    g.bench_function("16_digit", |b| {
        b.iter(|| {
            let mut buf = [0u8; 20];
            encode_fix_int(black_box(1_234_567_890_123_456), black_box(&mut buf))
        })
    });

    g.bench_function("negative_8", |b| {
        b.iter(|| {
            let mut buf = [0u8; 20];
            encode_fix_int(black_box(-12_345_678), black_box(&mut buf))
        })
    });

    g.finish();
}

fn bench_encode_uint(c: &mut Criterion) {
    c.bench_function("encode_fix_uint", |b| {
        b.iter(|| {
            let mut buf = [0u8; 10];
            encode_fix_uint(black_box(256), black_box(&mut buf))
        })
    });
}

fn bench_encode_seqnum(c: &mut Criterion) {
    c.bench_function("encode_fix_seqnum", |b| {
        b.iter(|| {
            let mut buf = [0u8; 20];
            encode_fix_seqnum(black_box(1_000_000), black_box(&mut buf))
        })
    });
}

fn bench_encode_timestamp(c: &mut Criterion) {
    let mut g = c.benchmark_group("FixTimestamp::encode");

    let ts_no_frac = FixTimestamp::parse(b"20260602-14:30:00").unwrap();
    let ts_millis = FixTimestamp::parse(b"20260602-14:30:00.123").unwrap();
    let ts_nanos = FixTimestamp::parse(b"20260602-14:30:00.123456789").unwrap();

    g.bench_function("no_frac", |b| {
        b.iter(|| {
            let mut buf = [0u8; 27];
            black_box(ts_no_frac).encode(black_box(&mut buf))
        })
    });

    g.bench_function("millis", |b| {
        b.iter(|| {
            let mut buf = [0u8; 27];
            black_box(ts_millis).encode(black_box(&mut buf))
        })
    });

    g.bench_function("nanos", |b| {
        b.iter(|| {
            let mut buf = [0u8; 27];
            black_box(ts_nanos).encode(black_box(&mut buf))
        })
    });

    g.finish();
}

fn bench_encode_date(c: &mut Criterion) {
    let date = FixDate::parse(b"20260602").unwrap();
    c.bench_function("FixDate::encode", |b| {
        b.iter(|| {
            let mut buf = [0u8; 8];
            black_box(date).encode(black_box(&mut buf))
        })
    });
}

fn bench_encode_time(c: &mut Criterion) {
    let mut g = c.benchmark_group("FixTime::encode");

    let time_no_frac = FixTime::parse(b"14:30:00").unwrap();
    let time_micros = FixTime::parse(b"14:30:00.123456").unwrap();

    g.bench_function("no_frac", |b| {
        b.iter(|| {
            let mut buf = [0u8; 18];
            black_box(time_no_frac).encode(black_box(&mut buf))
        })
    });

    g.bench_function("micros", |b| {
        b.iter(|| {
            let mut buf = [0u8; 18];
            black_box(time_micros).encode(black_box(&mut buf))
        })
    });

    g.finish();
}

// -- New domain-type benchmarks (parse + encode) --

fn bench_fix_char(c: &mut Criterion) {
    let mut g = c.benchmark_group("char");
    g.bench_function("parse", |b| b.iter(|| parse_fix_char(black_box(b"D"))));
    let ch = parse_fix_char(b"D").unwrap();
    g.bench_function("encode", |b| b.iter(|| encode_fix_char(black_box(ch))));
    g.finish();
}

fn bench_fix_text(c: &mut Criterion) {
    let mut g = c.benchmark_group("text");
    g.bench_function("parse_symbol", |b| {
        b.iter(|| parse_fix_text(black_box(b"BTC-USD")))
    });
    let t = parse_fix_text(b"BTC-USD").unwrap();
    g.bench_function("encode_symbol", |b| {
        b.iter(|| {
            let mut buf = [0u8; 16];
            encode_fix_text(black_box(t), black_box(&mut buf))
        })
    });
    g.finish();
}

fn bench_fix_month_year(c: &mut Criterion) {
    let mut g = c.benchmark_group("FixMonthYear");
    g.bench_function("parse_year_month", |b| {
        b.iter(|| FixMonthYear::parse(black_box(b"202603")))
    });
    g.bench_function("parse_year_month_day", |b| {
        b.iter(|| FixMonthYear::parse(black_box(b"20260318")))
    });
    g.bench_function("parse_year_month_week", |b| {
        b.iter(|| FixMonthYear::parse(black_box(b"202603w3")))
    });
    let my = FixMonthYear::parse(b"202603w3").unwrap();
    g.bench_function("encode", |b| {
        b.iter(|| {
            let mut buf = [0u8; 8];
            black_box(my).encode(black_box(&mut buf))
        })
    });
    g.finish();
}

fn bench_fix_tenor(c: &mut Criterion) {
    let mut g = c.benchmark_group("FixTenor");
    g.bench_function("parse_short", |b| {
        b.iter(|| FixTenor::parse(black_box(b"M3")))
    });
    g.bench_function("parse_multi_digit", |b| {
        b.iter(|| FixTenor::parse(black_box(b"D365")))
    });
    let t = FixTenor::parse(b"D365").unwrap();
    g.bench_function("encode", |b| {
        b.iter(|| {
            let mut buf = [0u8; 11];
            black_box(t).encode(black_box(&mut buf))
        })
    });
    g.finish();
}

fn bench_fix_tz(c: &mut Criterion) {
    let mut g = c.benchmark_group("FixTz");
    g.bench_function("tz_time_parse", |b| {
        b.iter(|| FixTzTime::parse(black_box(b"14:30:00.123-05:30")))
    });
    g.bench_function("tz_timestamp_parse", |b| {
        b.iter(|| FixTzTimestamp::parse(black_box(b"20260602-14:30:00+01:00")))
    });
    let tt = FixTzTime::parse(b"14:30:00.123-05:30").unwrap();
    g.bench_function("tz_time_encode", |b| {
        b.iter(|| {
            let mut buf = [0u8; 24];
            black_box(tt).encode(black_box(&mut buf))
        })
    });
    let ts = FixTzTimestamp::parse(b"20260602-14:30:00+01:00").unwrap();
    g.bench_function("tz_timestamp_encode", |b| {
        b.iter(|| {
            let mut buf = [0u8; 33];
            black_box(ts).encode(black_box(&mut buf))
        })
    });
    g.finish();
}

fn bench_multi_value(c: &mut Criterion) {
    let mut g = c.benchmark_group("MultipleValue");
    g.bench_function("multi_char", |b| {
        b.iter(|| parse_fix_multi_char(black_box(b"A B C D")).unwrap().count())
    });
    g.bench_function("multi_string", |b| {
        b.iter(|| {
            parse_fix_multi_string(black_box(b"FOO BAR BAZ"))
                .unwrap()
                .count()
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_fix_decimal,
    bench_fix_int,
    bench_fix_uint,
    bench_fix_seqnum,
    bench_fix_timestamp,
    bench_fix_date,
    bench_fix_time,
    bench_encode_decimal,
    bench_encode_int,
    bench_encode_uint,
    bench_encode_seqnum,
    bench_encode_timestamp,
    bench_encode_date,
    bench_encode_time,
    bench_fix_char,
    bench_fix_text,
    bench_fix_month_year,
    bench_fix_tenor,
    bench_fix_tz,
    bench_multi_value,
);
criterion_main!(benches);
